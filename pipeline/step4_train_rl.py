"""
Step 4: Offline RL training — Reward-Filtered SFT with LoRA.

Takes filtered dialogues (from step3) and trains model on the
"good" rounds (IG > 0). Redundant rounds have already been removed.

For rounds that remain:
  - IG > 0: train with positive weight (reinforce)
  - IG = 0 (neutral): train with weight 1.0 (standard SFT)

Usage:
    conda activate lx_vllm
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/mnt/data/.cache/huggingface
    python step4_train_rl.py data/  # train on all filtered episodes in data/

Output: output/lora_adapter/
"""
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

import config

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "output/lora_adapter"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_SEQ_LEN = 2048


def load_training_data(data_dir):
    """
    Load all filtered episodes and convert to training examples.
    Each example: (context, response, weight)
    - context = dialogue up to this round
    - response = winner agent's speech in this round
    - weight = info_gain (higher = more important to learn)
    """
    filtered_files = sorted(glob.glob(os.path.join(data_dir, "*_filtered.json")))

    if not filtered_files:
        print(f"No filtered files found in {data_dir}")
        return []

    examples = []
    for fpath in filtered_files:
        with open(fpath, "r") as f:
            data = json.load(f)

        winner = data["winner_name"]
        rounds = data["filtered_dialogue"]

        for k, r in enumerate(rounds):
            # Find winner's speech in this round
            winner_speech = None
            for msg in r["agent_messages"]:
                if msg["sender"] == winner:
                    if "did nothing" not in msg["content"] and "left the conversation" not in msg["content"]:
                        winner_speech = msg["content"]
                        break

            if not winner_speech:
                continue

            # Context = all previous rounds' dialogue
            context_parts = [prev["dialogue_text"] for prev in rounds[:k] if prev["dialogue_text"]]
            context = "\n".join(context_parts) if context_parts else "(conversation start)"

            # Weight based on info gain
            ig = r.get("info_gain", 0)
            weight = max(ig, 0.5)  # floor at 0.5 so neutral rounds still train

            examples.append({
                "context": context,
                "response": f"{winner}: {winner_speech}",
                "weight": weight,
                "source": os.path.basename(fpath),
            })

    print(f"Loaded {len(examples)} training examples from {len(filtered_files)} episodes")
    return examples


def compute_loss(model, tokenizer, context, response, weight, device):
    """Weighted SFT loss for one example."""
    full_text = f"{context}\n{response}"
    context_text = f"{context}\n"

    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    context_enc = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

    context_len = context_enc["input_ids"].shape[1]
    full_len = full_enc["input_ids"].shape[1]

    if full_len <= context_len:
        return None

    outputs = model(**full_enc)
    logits = outputs.logits

    shift_logits = logits[:, context_len - 1:full_len - 1, :]
    shift_labels = full_enc["input_ids"][:, context_len:full_len]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits.squeeze(0), shift_labels.squeeze(0))

    # Weight by info gain
    return weight * loss


def main(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

    print("Adding LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\nLoading data from: {data_dir}")
    examples = load_training_data(data_dir)

    if not examples:
        print("No training examples. Run step1-step3 first.")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print(f"\nTraining: {NUM_EPOCHS} epochs, {len(examples)} examples/epoch")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        steps = 0

        for i, ex in enumerate(examples):
            optimizer.zero_grad()
            loss = compute_loss(model, tokenizer, ex["context"], ex["response"], ex["weight"], device)

            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if (i + 1) % 5 == 0 or i == len(examples) - 1:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Step {i+1}/{len(examples)} | loss={loss.item():.4f}")

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch+1} done | avg_loss={avg_loss:.4f} | steps={steps}")
        print("-" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.DATA_DIR
    main(data_dir)
