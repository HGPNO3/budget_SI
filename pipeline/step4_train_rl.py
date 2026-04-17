"""
Step 4: Minimal RL training with TRL + LoRA.

Uses info_gain data from step2 as reward signal.
Offline REINFORCE: reinforce useful turns, penalize redundant ones.

Usage:
    conda activate lx_vllm
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/mnt/data/.cache/huggingface
    python step4_train_rl.py data/episode_XXXX_info_gain.json

Output: trained LoRA adapter saved to output/lora_adapter/
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "output/lora_adapter"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_SEQ_LEN = 2048


def load_training_data(info_gain_path):
    """
    Load info_gain JSON and convert to training examples.
    Each example: (context, response, reward)
    - context = dialogue up to this turn
    - response = this turn's content
    - reward = info gain (positive = useful, negative = redundant)
    """
    with open(info_gain_path, "r") as f:
        data = json.load(f)

    turns = data["turns"]
    examples = []

    for k, turn in enumerate(turns):
        sender = turn.get("sender", "")
        content = turn["content"]
        info_gain = turn["info_gain"]

        # Skip Environment turns and "did nothing" — they're not real dialogue
        if sender == "Environment":
            continue
        if "did nothing" in content or "left the conversation" in content:
            continue
        # Skip neutral turns (IG=0) — no learning signal
        if info_gain == 0:
            continue

        # Build context from all previous turns (only agent turns)
        context_parts = []
        for prev in turns[:k]:
            prev_sender = prev.get("sender", "")
            if prev_sender == "Environment":
                continue
            if "did nothing" in prev["content"]:
                continue
            context_parts.append(f"{prev_sender}: {prev['content']}")

        context = "\n".join(context_parts) if context_parts else "(conversation start)"
        response = f"{sender}: {content}"

        examples.append({
            "context": context,
            "response": response,
            "reward": info_gain,
        })

    print(f"Loaded {len(examples)} training examples:")
    useful = sum(1 for e in examples if e["reward"] > 0)
    redundant = sum(1 for e in examples if e["reward"] < 0)
    print(f"  Useful (reward > 0): {useful}")
    print(f"  Redundant (reward < 0): {redundant}")

    return examples


def compute_loss(model, tokenizer, context, response, reward, device):
    """
    REINFORCE loss for one example.
    loss = -reward * mean(log_prob(response_tokens | context))

    If reward > 0: minimizes loss by INCREASING response probability (reinforce)
    If reward < 0: minimizes loss by DECREASING response probability (penalize)
    """
    # Build full text: context + response
    full_text = f"{context}\n{response}"
    context_text = f"{context}\n"

    # Tokenize
    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    context_enc = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

    context_len = context_enc["input_ids"].shape[1]
    full_len = full_enc["input_ids"].shape[1]

    # If response is empty after context, skip
    if full_len <= context_len:
        return None

    # Forward pass
    outputs = model(**full_enc)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log probs for response tokens only
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, context_len - 1:full_len - 1, :]  # (1, resp_len, vocab)
    shift_labels = full_enc["input_ids"][:, context_len:full_len]  # (1, resp_len)

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, resp_len)

    # Mean log prob of response
    mean_log_prob = token_log_probs.mean()

    # REINFORCE loss: -reward * log_prob
    loss = -reward * mean_log_prob

    return loss


def main(info_gain_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 1. Load model + tokenizer
    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Add LoRA
    print("Adding LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load training data
    print(f"\nLoading data from: {info_gain_path}")
    examples = load_training_data(info_gain_path)

    if not examples:
        print("No training examples found. Exiting.")
        return

    # 4. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print(f"\nStarting training: {NUM_EPOCHS} epochs, {len(examples)} examples per epoch")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        valid_steps = 0

        for i, example in enumerate(examples):
            optimizer.zero_grad()

            loss = compute_loss(
                model, tokenizer,
                example["context"],
                example["response"],
                example["reward"],
                device,
            )

            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_steps += 1

            reward_str = f"+{example['reward']:.1f}" if example['reward'] > 0 else f"{example['reward']:.1f}"
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Step {i+1}/{len(examples)} | "
                  f"loss={loss.item():+.4f} | reward={reward_str}")

        avg_loss = total_loss / max(valid_steps, 1)
        print(f"Epoch {epoch+1} done | avg_loss={avg_loss:+.4f} | steps={valid_steps}")
        print("-" * 60)

    # 5. Save LoRA adapter
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")
    print("Done! To use: load base model + merge this adapter.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        import glob
        print("Usage: python step4_train_rl.py data/episode_XXXX_info_gain.json")
        print("\nAvailable info_gain files:")
        for f in sorted(glob.glob("data/*_info_gain.json")):
            print(f"  {f}")
        sys.exit(1)

    main(sys.argv[1])
