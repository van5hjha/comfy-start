#!/usr/bin/env bash
set -e

echo "===================================================================="
echo " Starting Custom RunPod ComfyUI Tier 2 Face Swap"
echo "===================================================================="

COMFY_DIR="/app/ComfyUI"
STORAGE_DIR="/workspace"

# If a persistent volume is mounted at /workspace, store/symlink models there
if [ -d "${STORAGE_DIR}" ]; then
    echo "[INFO] Persistent storage detected at /workspace"
    MODELS_BASE="${STORAGE_DIR}/models"
    mkdir -p "${MODELS_BASE}/text_encoders" \
             "${MODELS_BASE}/clip" \
             "${MODELS_BASE}/diffusion_models" \
             "${MODELS_BASE}/unet" \
             "${MODELS_BASE}/vae" \
             "${MODELS_BASE}/loras"

    # Link /workspace/models to ComfyUI/models if not already linked
    for d in text_encoders clip diffusion_models unet vae loras; do
        if [ ! -L "${COMFY_DIR}/models/${d}" ]; then
            rm -rf "${COMFY_DIR}/models/${d}"
            ln -sf "${MODELS_BASE}/${d}" "${COMFY_DIR}/models/${d}"
        fi
    done
else
    MODELS_BASE="${COMFY_DIR}/models"
fi

# Function to download model if missing or too small
download_model() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"
    mkdir -p "${dest_dir}"
    local filepath="${dest_dir}/${filename}"

    if [ -f "${filepath}" ] && [ $(stat -c%s "${filepath}" 2>/dev/null || stat -f%z "${filepath}" 2>/dev/null || echo 0) -gt 50000000 ]; then
        echo "  [EXISTS] ${filename}"
    else
        echo "  [DOWNLOADING] ${filename} via aria2..."
        aria2c -c -x 16 -s 16 -k 1M --file-allocation=none -d "${dest_dir}" -o "${filename}" "${url}" || \
        curl -L -C - -o "${filepath}" "${url}"
        echo "  [DONE] ${filename}"
    fi
}

echo "--- Verifying Tier 2 Model Weights ---"

# 1. Text Encoder: Qwen 3 8B FP8 Mixed
download_model \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
    "${MODELS_BASE}/text_encoders" \
    "qwen_3_8b_fp8mixed.safetensors"
ln -sf "${MODELS_BASE}/text_encoders/qwen_3_8b_fp8mixed.safetensors" "${MODELS_BASE}/clip/qwen_3_8b_fp8mixed.safetensors"

# 2. Diffusion Model: FLUX.2 Klein 9B FP8
download_model \
    "https://huggingface.co/MIUProject/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" \
    "${MODELS_BASE}/diffusion_models" \
    "flux-2-klein-9b.safetensors"
ln -sf "${MODELS_BASE}/diffusion_models/flux-2-klein-9b.safetensors" "${MODELS_BASE}/unet/flux-2-klein-9b.safetensors"
ln -sf "${MODELS_BASE}/diffusion_models/flux-2-klein-9b.safetensors" "${MODELS_BASE}/diffusion_models/flux-2-klein-9b-fp8.safetensors"
ln -sf "${MODELS_BASE}/diffusion_models/flux-2-klein-9b.safetensors" "${MODELS_BASE}/unet/flux-2-klein-9b-fp8.safetensors"

# 3. VAE: Flux 2 VAE
download_model \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "${MODELS_BASE}/vae" \
    "flux2-vae.safetensors"

# 4. LoRA: BFS Best Face Swap
download_model \
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors" \
    "${MODELS_BASE}/loras" \
    "Alissonerdx__BFS-Best-Face-Swap__bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors"

# 5. LoRA: Hyper-FLUX 8-Step LoRA
download_model \
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors" \
    "${MODELS_BASE}/loras" \
    "Hyper-FLUX.1-dev-8steps-lora.safetensors"

echo "===================================================================="
echo " All Tier 2 Models Ready! Launching ComfyUI on Port 8188..."
echo "===================================================================="

cd "${COMFY_DIR}"
exec python main.py --listen 0.0.0.0 --port 8188 --preview-method auto
