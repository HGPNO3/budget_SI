"""Server configuration. Adjust these to match your deployment."""

VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_MODEL_NAME = "gpt-4o"  # --served-model-name in vLLM
REDIS_URL = "redis://localhost:6380"
OPENAI_API_KEY = "EMPTY"

DATA_DIR = "data"  # where episode JSONs are saved
