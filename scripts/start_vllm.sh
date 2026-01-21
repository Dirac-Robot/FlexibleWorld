#!/bin/bash
# =============================================================================
# vLLM Server Startup Script for B200 8GPU
# =============================================================================
#
# Usage:
#   ./scripts/start_vllm.sh                    # Default: Qwen2.5-72B, 4 GPU
#   ./scripts/start_vllm.sh qwen72b            # Qwen2.5-72B-Instruct
#   ./scripts/start_vllm.sh llama70b           # Llama-3.1-70B-Instruct
#   ./scripts/start_vllm.sh qwen32b            # Qwen2.5-32B-Instruct (faster)
#
# =============================================================================

set -e

# Storage paths - ALL on /workspace to avoid disk issues
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
export VLLM_CACHE="/workspace/.cache/vllm"

# Create cache directories
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE"

# Default settings
MODEL_PRESET="${1:-qwen72b}"
PORT="${2:-8000}"

case "$MODEL_PRESET" in
    "qwen72b")
        MODEL="Qwen/Qwen2.5-72B-Instruct"
        TP_SIZE=4  # 4 GPU for 72B
        MAX_MODEL_LEN=8192
        ;;
    "qwen32b")
        MODEL="Qwen/Qwen2.5-32B-Instruct"
        TP_SIZE=2  # 2 GPU for 32B
        MAX_MODEL_LEN=16384
        ;;
    "qwen14b")
        MODEL="Qwen/Qwen2.5-14B-Instruct"
        TP_SIZE=1  # 1 GPU for 14B
        MAX_MODEL_LEN=32768
        ;;
    "llama70b")
        MODEL="meta-llama/Llama-3.1-70B-Instruct"
        TP_SIZE=4
        MAX_MODEL_LEN=8192
        ;;
    "llama8b")
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        TP_SIZE=1
        MAX_MODEL_LEN=32768
        ;;
    "qwen_vl")
        MODEL="Qwen/Qwen2-VL-72B-Instruct"
        TP_SIZE=4
        MAX_MODEL_LEN=4096
        ;;
    *)
        echo "Unknown preset: $MODEL_PRESET"
        echo "Available: qwen72b, qwen32b, qwen14b, llama70b, llama8b, qwen_vl"
        exit 1
        ;;
esac

echo "============================================"
echo "Starting vLLM Server"
echo "============================================"
echo "Model: $MODEL"
echo "Tensor Parallel: $TP_SIZE GPUs"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Port: $PORT"
echo "Cache Dir: $HF_HOME"
echo "============================================"

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "============================================"

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --trust-remote-code \
    --download-dir "$HF_HOME/hub" \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --enable-prefix-caching
