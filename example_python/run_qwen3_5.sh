SCRIPT_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR}

# Dependencies:
# uv venv qwen_env
# uv pip install transformers torch torchvision
# uv pip install accelerate

source ./qwen_env/bin/activate

# Default to using 2 GPUs: 0 and 1 (override by setting CUDA_VISIBLE_DEVICES).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Refer: https://huggingface.co/Qwen/Qwen3.5-122B-A10B#using-qwen35-via-the-chat-completions-api

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

# Optional overrides:
#   MODEL_ID=../models/Qwen/Qwen3-VL-8B-Instruct IMAGE_PATH=mathv-1327.jpg ./run_qwen3_5.sh

python example_qwen3_5.py
