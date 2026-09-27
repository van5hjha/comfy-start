#!/usr/bin/env bash
set -e

# ==============================================================================
# RunPod High-Speed Setup Script for ComfyUI Tier 2 Face Swap
# (FLUX.2 Klein 9B + BFS LoRA + Unstretched ReferenceLatents + 8-Step Morphology)
# ==============================================================================

echo "===================================================================="
echo " Starting RunPod Automated Setup for Tier 2 Face Swap"
echo "===================================================================="

WORKSPACE="/workspace"
COMFY_DIR="/root/ComfyUI"
MODELS_DIR="${WORKSPACE}/models"
OUTPUTS_DIR="${WORKSPACE}/output"
INPUTS_DIR="${WORKSPACE}/input"
WORKFLOWS_DIR="${WORKSPACE}/workflows"

# Clean up any broken leftover ComfyUI directory on NFS volume
rm -rf "${WORKSPACE}/ComfyUI" 2>/dev/null || true
mkdir -p "${WORKSPACE}" "${MODELS_DIR}" "${OUTPUTS_DIR}" "${INPUTS_DIR}" "${WORKFLOWS_DIR}"

# 1. System packages (aria2 for 10Gbps parallel downloads)
echo "[1/6] Installing system tools (aria2, git, curl)..."
apt-get update -qq && apt-get install -y -qq aria2 curl wget git

# 2. Setup ComfyUI on fast local container SSD (bypasses all NFS chmod/chown issues)
if [ ! -d "${COMFY_DIR}/.git" ]; then
    echo "[2/6] Cloning ComfyUI to local container disk (${COMFY_DIR})..."
    rm -rf "${COMFY_DIR}"
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFY_DIR}"
else
    echo "[2/6] ComfyUI already exists in ${COMFY_DIR}."
fi

cd "${COMFY_DIR}"

# 3. Python environment & requirements
echo "[3/6] Installing ComfyUI core requirements..."
pip install -q -r requirements.txt

# Ensure PyTorch is >= 2.5 for native GQA (Qwen 3 text encoder) and modern kernel support
if python -c "import torch; exit(0 if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 5) else 1)" 2>/dev/null; then
    echo "  [INFO] PyTorch is already >= 2.5 ($(python -c 'import torch; print(torch.__version__)'))."
else
    if python -c "import torch; exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12 else 1)" 2>/dev/null; then
        echo "  [INFO] NVIDIA Blackwell GPU (sm_120) detected. Installing PyTorch with CUDA 12.8..."
        pip install -q --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    else
        echo "  [INFO] Upgrading PyTorch to >= 2.5 with CUDA 12.4 for native GQA and FlashAttention support..."
        pip install -q --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi
fi

pip install -q --force-reinstall --no-deps comfy-kitchen

# 4. Setup custom nodes (ComfyUI-Manager and ComfyUI-KJNodes) on local SSD
echo "[4/6] Setting up custom nodes..."
CUSTOM_NODES="${COMFY_DIR}/custom_nodes"
mkdir -p "${CUSTOM_NODES}"

if [ ! -d "${CUSTOM_NODES}/ComfyUI-Manager/.git" ]; then
    echo "  -> Cloning ComfyUI-Manager..."
    rm -rf "${CUSTOM_NODES}/ComfyUI-Manager"
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "${CUSTOM_NODES}/ComfyUI-Manager"
fi

if [ ! -d "${CUSTOM_NODES}/ComfyUI-KJNodes/.git" ]; then
    echo "  -> Cloning ComfyUI-KJNodes..."
    rm -rf "${CUSTOM_NODES}/ComfyUI-KJNodes"
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git "${CUSTOM_NODES}/ComfyUI-KJNodes"
fi

echo "  -> Installing custom node dependencies..."
pip install -q color-matcher mss opencv-python-headless GitPython
pip install -q -r "${CUSTOM_NODES}/ComfyUI-Manager/requirements.txt" 2>/dev/null || true
pip install -q -r "${CUSTOM_NODES}/ComfyUI-KJNodes/requirements.txt" 2>/dev/null || true

# Apply PyTorch 2.4 compatibility patches for comfy_kitchen and quant_ops
echo "  -> Applying PyTorch 2.4 compatibility patches..."
python - << 'PYEOF'
import os, sys, glob, site, re

# A. Patch ComfyUI quant_ops.py: catch (ImportError, Exception) so startup never crashes
for quant_path in glob.glob("/root/ComfyUI/comfy/quant_ops.py"):
    try:
        with open(quant_path, "r", encoding="utf-8") as f:
            code = f.read()
        if "except (ImportError, Exception) as e:" not in code:
            code = code.replace("except ImportError as e:", "except (ImportError, Exception) as e:")
            with open(quant_path, "w", encoding="utf-8") as f:
                f.write(code)
            print("  ✓ Patched comfy/quant_ops.py for robust error handling")
    except Exception as e:
        print(f"  Warning patching quant_ops: {e}")

# B. Patch torch._library.infer_schema.py: support string annotations, Python 3.10 unions, and list[...]
for sp in site.getsitepackages():
    for p in glob.glob(os.path.join(sp, "torch", "_library", "infer_schema.py")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            if "PATCH_FLEXIBLE_SCHEMA_SUPPORT" not in content:
                patch = '''
# --- PATCH_FLEXIBLE_SCHEMA_SUPPORT ---
try:
    import types as _types, typing as _typing, torch as _torch
    _str_param_map = {
        "torch.Tensor": "Tensor", "Tensor": "Tensor",
        "float": "float", "int": "int", "bool": "bool", "str": "str",
        "float | None": "float?", "int | None": "int?", "bool | None": "bool?", "str | None": "str?",
        "torch.Tensor | None": "Tensor?", "Tensor | None": "Tensor?",
        "Optional[torch.Tensor]": "Tensor?", "Optional[Tensor]": "Tensor?",
        "Optional[float]": "float?", "Optional[int]": "int?", "Optional[bool]": "bool?",
        "list[int]": "int[]", "List[int]": "int[]",
        "list[float]": "float[]", "List[float]": "float[]",
        "list[torch.Tensor]": "Tensor[]", "List[torch.Tensor]": "Tensor[]",
        "list[Tensor]": "Tensor[]", "List[Tensor]": "Tensor[]",
        "torch.dtype": "ScalarType", "torch.device": "Device",
    }
    SUPPORTED_PARAM_TYPES.update(_str_param_map)
    SUPPORTED_RETURN_TYPES.update({
        "torch.Tensor": "Tensor", "Tensor": "Tensor", "None": "()",
        "list[torch.Tensor]": "Tensor[]", "List[torch.Tensor]": "Tensor[]",
        "list[Tensor]": "Tensor[]", "List[Tensor]": "Tensor[]",
    })
    for _k, _v in list(SUPPORTED_PARAM_TYPES.items()):
        if getattr(_k, "__origin__", None) is list:
            _args = getattr(_k, "__args__", None)
            if _args:
                SUPPORTED_PARAM_TYPES[list[_args]] = _v
    for _k, _v in list(SUPPORTED_RETURN_TYPES.items()):
        if getattr(_k, "__origin__", None) is list:
            _args = getattr(_k, "__args__", None)
            if _args:
                SUPPORTED_RETURN_TYPES[list[_args]] = _v

    class _FlexibleDict(dict):
        def __contains__(self, key):
            if super().__contains__(key):
                return True
            kstr = str(key).strip().replace("typing.", "").replace("torch.", "")
            if super().__contains__(kstr):
                return True
            for k in list(self.keys()):
                if str(k) == str(key) or str(k).replace("typing.", "").replace("torch.", "") == kstr:
                    return True
            if "Tensor" in kstr or "float" in kstr or "int" in kstr or "bool" in kstr:
                return True
            return False

        def __getitem__(self, key):
            try:
                return super().__getitem__(key)
            except KeyError:
                kstr = str(key).strip().replace("typing.", "").replace("torch.", "")
                if super().__contains__(kstr):
                    return super().__getitem__(kstr)
                for k in list(self.keys()):
                    if str(k) == str(key) or str(k).replace("typing.", "").replace("torch.", "") == kstr:
                        return super().__getitem__(k)
                if "Tensor" in kstr:
                    if "None" in kstr or "Optional" in kstr or "?" in kstr:
                        return "Tensor?"
                    if "list" in kstr.lower() or "sequence" in kstr.lower():
                        return "Tensor[]"
                    return "Tensor"
                if "float" in kstr:
                    return "float?" if ("None" in kstr or "Optional" in kstr) else "float"
                if "int" in kstr:
                    return "int?" if ("None" in kstr or "Optional" in kstr) else "int"
                if "bool" in kstr:
                    return "bool?" if ("None" in kstr or "Optional" in kstr) else "bool"
                if "None" in kstr:
                    return "()"
                return "Tensor"

        def keys(self):
            return self

    SUPPORTED_PARAM_TYPES = _FlexibleDict(SUPPORTED_PARAM_TYPES)
    SUPPORTED_RETURN_TYPES = _FlexibleDict(SUPPORTED_RETURN_TYPES)
except Exception as _e:
    pass
# -------------------------------------
'''
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content + "\n" + patch)
                print(f"  ✓ Patched PyTorch infer_schema.py ({p})")
        except Exception as e:
            print(f"  Warning patching infer_schema: {e}")

# C. Patch comfy_kitchen custom operator files to supply explicit schemas & fix type hints
for sp in site.getsitepackages():
    for p in glob.glob(os.path.join(sp, "comfy_kitchen", "**", "*.py"), recursive=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            modified = False

            # 1. sage_attention.py: provide explicit schema for int8_attention
            if "int8_attention" in content and "schema=" not in content:
                content = content.replace(
                    '@torch.library.custom_op("comfy_kitchen::int8_attention", mutates_args=())',
                    '@torch.library.custom_op("comfy_kitchen::int8_attention", mutates_args=(), schema="(Tensor q, Tensor k, Tensor v, float? scale=None) -> Tensor")'
                )
                modified = True

            # 2. conv3d.py: provide explicit schema for fp16_conv3d
            if "fp16_conv3d" in content and "schema=" not in content:
                content = content.replace(
                    '@torch.library.custom_op("comfy_kitchen::fp16_conv3d", mutates_args=())',
                    '@torch.library.custom_op("comfy_kitchen::fp16_conv3d", mutates_args=(), schema="(Tensor input, Tensor weight, Tensor? bias=None, int[] stride=[1, 1, 1], int[] padding=[0, 0, 0], int[] dilation=[1, 1, 1], int groups=1) -> Tensor")'
                )
                modified = True

            # 3. Replace any list[...] with typing.List[...] in custom_op files
            if "custom_op" in content and "list[" in content:
                if "import typing\n" not in content and "from typing import" not in content:
                    content = "import typing\n" + content
                content = re.sub(r"\blist\[([a-zA-Z0-9_\.]+)\]", r"typing.List[\1]", content)
                modified = True

            if modified:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  ✓ Patched custom_op file: {os.path.basename(p)}")
        except Exception as e:
            print(f"  Warning patching comfy_kitchen file {p}: {e}")

# D. Patch ComfyUI ops.py: support enable_gqa on all PyTorch versions (< 2.5 fallback)
for ops_path in glob.glob("/root/ComfyUI/comfy/ops.py"):
    try:
        with open(ops_path, "r", encoding="utf-8") as f:
            code = f.read()
        if "PATCH_SAFE_SDPA_GQA" not in code:
            patch = '''
# --- PATCH_SAFE_SDPA_GQA ---
try:
    import torch
    _orig_torch_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _safe_sdpa(q, k, v, *args, **kwargs):
        if "enable_gqa" in kwargs:
            gqa = kwargs.pop("enable_gqa")
            try:
                return _orig_torch_sdpa(q, k, v, *args, enable_gqa=gqa, **kwargs)
            except TypeError:
                if k.size(-3) != q.size(-3):
                    repeats = q.size(-3) // k.size(-3)
                    k = k.repeat_interleave(repeats, dim=-3)
                    v = v.repeat_interleave(repeats, dim=-3)
                return _orig_torch_sdpa(q, k, v, *args, **kwargs)
        return _orig_torch_sdpa(q, k, v, *args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = _safe_sdpa
except Exception:
    pass
# ---------------------------
'''
            code = code + "\n" + patch
            with open(ops_path, "w", encoding="utf-8") as f:
                f.write(code)
            print("  ✓ Patched comfy/ops.py for safe GQA attention compatibility")
    except Exception as e:
        print(f"  Warning patching ops.py: {e}")
PYEOF

# 5. Link Persistent Network Volume (/workspace) for Models, Outputs, and Workflows
echo "[5/6] Linking persistent network volume (/workspace) to ComfyUI..."
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/clip"
mkdir -p "${MODELS_DIR}/diffusion_models"
mkdir -p "${MODELS_DIR}/unet"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/loras"

rm -rf "${COMFY_DIR}/models"
ln -sf "${MODELS_DIR}" "${COMFY_DIR}/models"

rm -rf "${COMFY_DIR}/output"
ln -sf "${OUTPUTS_DIR}" "${COMFY_DIR}/output"

rm -rf "${COMFY_DIR}/input"
ln -sf "${INPUTS_DIR}" "${COMFY_DIR}/input"

mkdir -p "${COMFY_DIR}/user/default"
rm -rf "${COMFY_DIR}/user/default/workflows"
ln -sf "${WORKFLOWS_DIR}" "${COMFY_DIR}/user/default/workflows"

# 6. Fast Parallel Model Downloads directly to Persistent Network Volume
echo "[6/6] Checking & downloading required model weights (~18 GB total)..."

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

# A. Qwen 3 8B Text Encoder
download_model \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
    "${MODELS_DIR}/text_encoders" \
    "qwen_3_8b_fp8mixed.safetensors"
ln -sf "${MODELS_DIR}/text_encoders/qwen_3_8b_fp8mixed.safetensors" "${MODELS_DIR}/clip/qwen_3_8b_fp8mixed.safetensors"

# B. FLUX.2 Klein 9B Diffusion Model
download_model \
    "https://huggingface.co/MIUProject/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors" \
    "${MODELS_DIR}/diffusion_models" \
    "flux-2-klein-9b.safetensors"
ln -sf "${MODELS_DIR}/diffusion_models/flux-2-klein-9b.safetensors" "${MODELS_DIR}/unet/flux-2-klein-9b.safetensors"
ln -sf "${MODELS_DIR}/diffusion_models/flux-2-klein-9b.safetensors" "${MODELS_DIR}/diffusion_models/flux-2-klein-9b-fp8.safetensors"
ln -sf "${MODELS_DIR}/diffusion_models/flux-2-klein-9b.safetensors" "${MODELS_DIR}/unet/flux-2-klein-9b-fp8.safetensors"

# C. FLUX 2 VAE
download_model \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "${MODELS_DIR}/vae" \
    "flux2-vae.safetensors"

# D. BFS Best Face Swap LoRA
download_model \
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors" \
    "${MODELS_DIR}/loras" \
    "Alissonerdx__BFS-Best-Face-Swap__bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors"

# E. Hyper-FLUX 8-Step LoRA
download_model \
    "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors" \
    "${MODELS_DIR}/loras" \
    "Hyper-FLUX.1-dev-8steps-lora.safetensors"

# Deploy Morphology-Accurate Workflow JSON directly into persistent workflows directory
cat << 'EOF' > "${WORKFLOWS_DIR}/face_swap_3ref_masked_workflow.json"
{
  "config": {},
  "definitions": {
    "subgraphs": [
      {
        "config": {},
        "extra": {
          "links_added_by_ue": [],
          "ue_links": [],
          "workflowRendererVersion": "LG"
        },
        "groups": [],
        "id": "6e5070f7-26e8-4a9a-ae4d-5d3fc1c591af",
        "inputNode": {
          "bounding": [
            -270,
            990,
            120,
            120
          ],
          "id": -10
        },
        "inputs": [
          {
            "id": "5c9a0f5e-8cee-4947-90bc-330de782043a",
            "label": "positive",
            "linkIds": [
              165
            ],
            "name": "conditioning",
            "pos": [
              -170,
              1010
            ],
            "type": "CONDITIONING"
          },
          {
            "id": "61826d46-4c21-4ad6-801c-3e3fa94115e2",
            "label": "negative",
            "linkIds": [
              166
            ],
            "name": "conditioning_1",
            "pos": [
              -170,
              1030
            ],
            "type": "CONDITIONING"
          },
          {
            "id": "345bf085-5939-47ff-9767-8f8f239a719c",
            "linkIds": [
              167
            ],
            "name": "pixels",
            "pos": [
              -170,
              1050
            ],
            "type": "IMAGE"
          },
          {
            "id": "f4594e34-e2f5-4f1e-b1fa-a1dc2aeb0a90",
            "linkIds": [
              168
            ],
            "name": "vae",
            "pos": [
              -170,
              1070
            ],
            "type": "VAE"
          }
        ],
        "links": [
          {
            "id": 163,
            "origin_id": 78,
            "origin_slot": 0,
            "target_id": 179,
            "target_slot": 1,
            "type": "LATENT"
          },
          {
            "id": 164,
            "origin_id": 78,
            "origin_slot": 0,
            "target_id": 77,
            "target_slot": 1,
            "type": "LATENT"
          },
          {
            "id": 165,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": 77,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 166,
            "origin_id": -10,
            "origin_slot": 1,
            "target_id": 179,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 167,
            "origin_id": -10,
            "origin_slot": 2,
            "target_id": 78,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 168,
            "origin_id": -10,
            "origin_slot": 3,
            "target_id": 78,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 169,
            "origin_id": 77,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 170,
            "origin_id": 179,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 1,
            "type": "CONDITIONING"
          }
        ],
        "name": "Reference Conditioning",
        "nodes": [
          {
            "flags": {
              "collapsed": false
            },
            "id": 179,
            "inputs": [
              {
                "link": 166,
                "localized_name": "conditioning",
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "link": 163,
                "localized_name": "latent",
                "name": "latent",
                "shape": 7,
                "type": "LATENT"
              }
            ],
            "mode": 0,
            "order": 2,
            "outputs": [
              {
                "links": [
                  170
                ],
                "localized_name": "CONDITIONING",
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              170,
              1050
            ],
            "properties": {
              "Node name for S&R": "ReferenceLatent",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              210,
              50
            ],
            "type": "ReferenceLatent",
            "widgets_values": []
          },
          {
            "flags": {
              "collapsed": false
            },
            "id": 78,
            "inputs": [
              {
                "link": 167,
                "localized_name": "pixels",
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 168,
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 1,
            "outputs": [
              {
                "links": [
                  163,
                  164
                ],
                "localized_name": "LATENT",
                "name": "LATENT",
                "type": "LATENT"
              }
            ],
            "pos": [
              -90,
              1150
            ],
            "properties": {
              "Node name for S&R": "VAEEncode",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              190,
              50
            ],
            "type": "VAEEncode",
            "widgets_values": []
          },
          {
            "flags": {
              "collapsed": false
            },
            "id": 77,
            "inputs": [
              {
                "link": 165,
                "localized_name": "conditioning",
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "link": 164,
                "localized_name": "latent",
                "name": "latent",
                "shape": 7,
                "type": "LATENT"
              }
            ],
            "mode": 0,
            "order": 0,
            "outputs": [
              {
                "links": [
                  169
                ],
                "localized_name": "CONDITIONING",
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              170,
              940
            ],
            "properties": {
              "Node name for S&R": "ReferenceLatent",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              210,
              50
            ],
            "type": "ReferenceLatent",
            "widgets_values": []
          }
        ],
        "outputNode": {
          "bounding": [
            580,
            970,
            120,
            80
          ],
          "id": -20
        },
        "outputs": [
          {
            "id": "b3357c0e-6428-4055-9cd3-3595f0896fa8",
            "label": "positive",
            "linkIds": [
              169
            ],
            "name": "CONDITIONING",
            "pos": [
              600,
              990
            ],
            "type": "CONDITIONING"
          },
          {
            "id": "01519713-2ed1-4694-a387-79f44e088e89",
            "label": "negative",
            "linkIds": [
              170
            ],
            "name": "CONDITIONING_1",
            "pos": [
              600,
              1010
            ],
            "type": "CONDITIONING"
          }
        ],
        "revision": 0,
        "state": {
          "lastGroupId": 7,
          "lastLinkId": 379,
          "lastNodeId": 206,
          "lastRerouteId": 4
        },
        "version": 1,
        "widgets": []
      },
      {
        "config": {},
        "extra": {
          "links_added_by_ue": [],
          "ue_links": [],
          "workflowRendererVersion": "LG"
        },
        "groups": [],
        "id": "5eb5cfa6-f140-40b7-bc6f-9477d677f888",
        "inputNode": {
          "bounding": [
            -270,
            990,
            120,
            120
          ],
          "id": -10
        },
        "inputs": [
          {
            "id": "5c9a0f5e-8cee-4947-90bc-330de782043a",
            "label": "positive",
            "linkIds": [
              165
            ],
            "name": "conditioning",
            "pos": [
              -170,
              1010
            ],
            "type": "CONDITIONING"
          },
          {
            "id": "61826d46-4c21-4ad6-801c-3e3fa94115e2",
            "label": "negative",
            "linkIds": [
              166
            ],
            "name": "conditioning_1",
            "pos": [
              -170,
              1030
            ],
            "type": "CONDITIONING"
          },
          {
            "id": "345bf085-5939-47ff-9767-8f8f239a719c",
            "linkIds": [
              167
            ],
            "name": "pixels",
            "pos": [
              -170,
              1050
            ],
            "type": "IMAGE"
          },
          {
            "id": "f4594e34-e2f5-4f1e-b1fa-a1dc2aeb0a90",
            "linkIds": [
              168
            ],
            "name": "vae",
            "pos": [
              -170,
              1070
            ],
            "type": "VAE"
          }
        ],
        "links": [
          {
            "id": 163,
            "origin_id": 181,
            "origin_slot": 0,
            "target_id": 180,
            "target_slot": 1,
            "type": "LATENT"
          },
          {
            "id": 164,
            "origin_id": 181,
            "origin_slot": 0,
            "target_id": 182,
            "target_slot": 1,
            "type": "LATENT"
          },
          {
            "id": 165,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": 182,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 166,
            "origin_id": -10,
            "origin_slot": 1,
            "target_id": 180,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 167,
            "origin_id": -10,
            "origin_slot": 2,
            "target_id": 181,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 168,
            "origin_id": -10,
            "origin_slot": 3,
            "target_id": 181,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 169,
            "origin_id": 182,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 170,
            "origin_id": 180,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 1,
            "type": "CONDITIONING"
          }
        ],
        "name": "Reference Conditioning",
        "nodes": [
          {
            "flags": {
              "collapsed": false
            },
            "id": 180,
            "inputs": [
              {
                "link": 166,
                "localized_name": "conditioning",
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "link": 163,
                "localized_name": "latent",
                "name": "latent",
                "shape": 7,
                "type": "LATENT"
              }
            ],
            "mode": 0,
            "order": 0,
            "outputs": [
              {
                "links": [
                  170
                ],
                "localized_name": "CONDITIONING",
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              170,
              1050
            ],
            "properties": {
              "Node name for S&R": "ReferenceLatent",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              210,
              50
            ],
            "type": "ReferenceLatent",
            "widgets_values": []
          },
          {
            "flags": {
              "collapsed": false
            },
            "id": 181,
            "inputs": [
              {
                "link": 167,
                "localized_name": "pixels",
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 168,
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 1,
            "outputs": [
              {
                "links": [
                  163,
                  164
                ],
                "localized_name": "LATENT",
                "name": "LATENT",
                "type": "LATENT"
              }
            ],
            "pos": [
              -90,
              1150
            ],
            "properties": {
              "Node name for S&R": "VAEEncode",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              190,
              50
            ],
            "type": "VAEEncode",
            "widgets_values": []
          },
          {
            "flags": {
              "collapsed": false
            },
            "id": 182,
            "inputs": [
              {
                "link": 165,
                "localized_name": "conditioning",
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "link": 164,
                "localized_name": "latent",
                "name": "latent",
                "shape": 7,
                "type": "LATENT"
              }
            ],
            "mode": 0,
            "order": 2,
            "outputs": [
              {
                "links": [
                  169
                ],
                "localized_name": "CONDITIONING",
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              170,
              940
            ],
            "properties": {
              "Node name for S&R": "ReferenceLatent",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              210,
              50
            ],
            "type": "ReferenceLatent",
            "widgets_values": []
          }
        ],
        "outputNode": {
          "bounding": [
            580,
            970,
            120,
            80
          ],
          "id": -20
        },
        "outputs": [
          {
            "id": "b3357c0e-6428-4055-9cd3-3595f0896fa8",
            "label": "positive",
            "linkIds": [
              169
            ],
            "name": "CONDITIONING",
            "pos": [
              600,
              990
            ],
            "type": "CONDITIONING"
          },
          {
            "id": "01519713-2ed1-4694-a387-79f44e088e89",
            "label": "negative",
            "linkIds": [
              170
            ],
            "name": "CONDITIONING_1",
            "pos": [
              600,
              1010
            ],
            "type": "CONDITIONING"
          }
        ],
        "revision": 0,
        "state": {
          "lastGroupId": 7,
          "lastLinkId": 379,
          "lastNodeId": 206,
          "lastRerouteId": 4
        },
        "version": 1,
        "widgets": []
      },
      {
        "config": {},
        "extra": {
          "reroutes": [
            {
              "id": 2,
              "linkIds": [
                349,
                350
              ],
              "pos": [
                370,
                4074
              ]
            },
            {
              "id": 4,
              "linkIds": [
                347,
                351,
                352,
                353
              ],
              "pos": [
                360,
                3874
              ]
            }
          ]
        },
        "groups": [
          {
            "bounding": [
              290,
              2380,
              1030,
              540
            ],
            "color": "#3f789e",
            "flags": {},
            "id": 7,
            "title": "LLM Expression Analysis"
          },
          {
            "bounding": [
              780,
              3160,
              460,
              570
            ],
            "color": "#3f789e",
            "flags": {},
            "id": 4,
            "title": "Prompt & Encoders"
          },
          {
            "bounding": [
              290,
              3160,
              380,
              550
            ],
            "color": "#3f789e",
            "flags": {},
            "id": 6,
            "title": "Flux Klein Models"
          },
          {
            "bounding": [
              -430,
              3160,
              1020,
              520
            ],
            "color": "#2e7d32",
            "flags": {},
            "id": 8,
            "title": "Kid A Multi-Reference Preprocessing"
          },
          {
            "bounding": [
              770,
              3780,
              280,
              710
            ],
            "color": "#8e24aa",
            "flags": {},
            "id": 9,
            "title": "4-Stage Reference Conditioning Cascade"
          },
          {
            "bounding": [
              430,
              2840,
              850,
              240
            ],
            "color": "#d81b60",
            "flags": {},
            "id": 10,
            "title": "Target Face Mask & Noise Latent"
          },
          {
            "bounding": [
              1430,
              3160,
              340,
              632
            ],
            "color": "#3f789e",
            "flags": {},
            "id": 5,
            "title": "4-Step Klein Sampler"
          },
          {
            "bounding": [
              2360,
              3160,
              270,
              330
            ],
            "color": "#00897b",
            "flags": {},
            "id": 11,
            "title": "Pristine Composite"
          }
        ],
        "id": "217f3726-59bc-4170-9ab1-c622bb578338",
        "inputNode": {
          "bounding": [
            -598,
            3211,
            168.25,
            290
          ],
          "id": -10
        },
        "inputs": [
          {
            "id": "4521d619-b84b-421c-8822-054e65cba061",
            "name": "noise_seed",
            "type": "INT",
            "linkIds": [
              370
            ],
            "pos": [
              -453.75,
              3235
            ]
          },
          {
            "id": "955c4b39-e6a2-403f-8295-755e2dae8453",
            "name": "lora_name",
            "type": "COMBO",
            "linkIds": [
              363
            ],
            "pos": [
              -453.75,
              3295
            ]
          },
          {
            "id": "3d6acd96-cd45-4156-89f3-723d2a7d638e",
            "name": "unet_name",
            "type": "COMBO",
            "linkIds": [
              369
            ],
            "pos": [
              -453.75,
              3315
            ]
          },
          {
            "id": "c028f281-6bb6-405e-8e28-c43e72e0e9d8",
            "name": "clip_name",
            "type": "COMBO",
            "linkIds": [
              368
            ],
            "pos": [
              -453.75,
              3335
            ]
          },
          {
            "id": "cfabd417-0d71-43cb-85fd-a470e44dd081",
            "name": "vae_name",
            "type": "COMBO",
            "linkIds": [
              362
            ],
            "pos": [
              -453.75,
              3355
            ]
          },
          {
            "id": "cc758fb6-c116-40df-bdf3-4e42ff0f0881",
            "name": "target_image",
            "localized_name": "target_image",
            "type": "IMAGE",
            "linkIds": [
              221
            ],
            "pos": [
              -453.75,
              3375
            ]
          },
          {
            "id": "a1b2c3d4-0001-40df-bdf3-4e42ff0f0881",
            "name": "target_mask",
            "localized_name": "target_mask",
            "type": "MASK",
            "linkIds": [
              416
            ],
            "pos": [
              -453.75,
              3395
            ]
          },
          {
            "id": "67274f8a-ff3a-41f0-b95f-269202052fc0",
            "name": "source_ref_1",
            "localized_name": "source_ref_1",
            "type": "IMAGE",
            "linkIds": [
              282
            ],
            "pos": [
              -453.75,
              3415
            ]
          },
          {
            "id": "a1b2c3d4-0002-40df-bdf3-4e42ff0f0881",
            "name": "source_ref_2",
            "localized_name": "source_ref_2",
            "type": "IMAGE",
            "linkIds": [
              401
            ],
            "pos": [
              -453.75,
              3435
            ]
          },
          {
            "id": "a1b2c3d4-0003-40df-bdf3-4e42ff0f0881",
            "name": "source_ref_3",
            "localized_name": "source_ref_3",
            "type": "IMAGE",
            "linkIds": [
              404
            ],
            "pos": [
              -453.75,
              3455
            ]
          },
          {
            "id": "a1b2c3d4-0004-40df-bdf3-4e42ff0f0881",
            "linkIds": [
              426
            ],
            "name": "source_ref_1_mask",
            "localized_name": "source_ref_1_mask",
            "pos": [
              -453.75,
              3475
            ],
            "type": "MASK"
          },
          {
            "id": "a1b2c3d4-0005-40df-bdf3-4e42ff0f0881",
            "linkIds": [
              427
            ],
            "name": "source_ref_2_mask",
            "localized_name": "source_ref_2_mask",
            "pos": [
              -453.75,
              3495
            ],
            "type": "MASK"
          },
          {
            "id": "a1b2c3d4-0006-40df-bdf3-4e42ff0f0881",
            "linkIds": [
              428
            ],
            "name": "source_ref_3_mask",
            "localized_name": "source_ref_3_mask",
            "pos": [
              -453.75,
              3515
            ],
            "type": "MASK"
          }
        ],
        "links": [
          {
            "id": 351,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 112,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 195,
            "origin_id": 116,
            "origin_slot": 0,
            "target_id": 114,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 196,
            "origin_id": 116,
            "origin_slot": 1,
            "target_id": 114,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 349,
            "origin_id": 121,
            "origin_slot": 0,
            "target_id": 234,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 198,
            "origin_id": 111,
            "origin_slot": 0,
            "target_id": 114,
            "target_slot": 3,
            "type": "VAE"
          },
          {
            "id": 199,
            "origin_id": 117,
            "origin_slot": 0,
            "target_id": 116,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 200,
            "origin_id": 119,
            "origin_slot": 0,
            "target_id": 116,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 347,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 116,
            "target_slot": 2,
            "type": "IMAGE"
          },
          {
            "id": 202,
            "origin_id": 111,
            "origin_slot": 0,
            "target_id": 116,
            "target_slot": 3,
            "type": "VAE"
          },
          {
            "id": 193,
            "origin_id": 112,
            "origin_slot": 0,
            "target_id": 113,
            "target_slot": 0,
            "type": "INT"
          },
          {
            "id": 194,
            "origin_id": 112,
            "origin_slot": 1,
            "target_id": 113,
            "target_slot": 1,
            "type": "INT"
          },
          {
            "id": 187,
            "origin_id": 110,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 0,
            "type": "NOISE"
          },
          {
            "id": 188,
            "origin_id": 123,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 1,
            "type": "GUIDER"
          },
          {
            "id": 189,
            "origin_id": 125,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 2,
            "type": "SAMPLER"
          },
          {
            "id": 310,
            "origin_id": 122,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 3,
            "type": "SIGMAS"
          },
          {
            "id": 204,
            "origin_id": 115,
            "origin_slot": 0,
            "target_id": 119,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 207,
            "origin_id": 133,
            "origin_slot": 0,
            "target_id": 123,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 353,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 176,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 216,
            "origin_id": 109,
            "origin_slot": 0,
            "target_id": 131,
            "target_slot": 0,
            "type": "LATENT"
          },
          {
            "id": 217,
            "origin_id": 111,
            "origin_slot": 0,
            "target_id": 131,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 304,
            "origin_id": 176,
            "origin_slot": 0,
            "target_id": 177,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 205,
            "origin_id": 112,
            "origin_slot": 0,
            "target_id": 122,
            "target_slot": 1,
            "type": "INT"
          },
          {
            "id": 206,
            "origin_id": 112,
            "origin_slot": 1,
            "target_id": 122,
            "target_slot": 2,
            "type": "INT"
          },
          {
            "id": 203,
            "origin_id": 115,
            "origin_slot": 0,
            "target_id": 117,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 344,
            "origin_id": 198,
            "origin_slot": 0,
            "target_id": 117,
            "target_slot": 1,
            "type": "STRING"
          },
          {
            "id": 317,
            "origin_id": 118,
            "origin_slot": 0,
            "target_id": 133,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 221,
            "origin_id": -10,
            "origin_slot": 5,
            "target_id": 120,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 282,
            "origin_id": -10,
            "origin_slot": 7,
            "target_id": 121,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 352,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 2,
            "type": "IMAGE"
          },
          {
            "id": 362,
            "origin_id": -10,
            "origin_slot": 4,
            "target_id": 111,
            "target_slot": 0,
            "type": "COMBO"
          },
          {
            "id": 363,
            "origin_id": -10,
            "origin_slot": 1,
            "target_id": 133,
            "target_slot": 1,
            "type": "COMBO"
          },
          {
            "id": 368,
            "origin_id": -10,
            "origin_slot": 3,
            "target_id": 115,
            "target_slot": 0,
            "type": "COMBO"
          },
          {
            "id": 369,
            "origin_id": -10,
            "origin_slot": 2,
            "target_id": 118,
            "target_slot": 0,
            "type": "COMBO"
          },
          {
            "id": 370,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": 110,
            "target_slot": 0,
            "type": "INT"
          },
          {
            "id": 416,
            "origin_id": -10,
            "origin_slot": 6,
            "target_id": 226,
            "target_slot": 0,
            "type": "MASK"
          },
          {
            "id": 401,
            "origin_id": -10,
            "origin_slot": 8,
            "target_id": 220,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 404,
            "origin_id": -10,
            "origin_slot": 9,
            "target_id": 221,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 407,
            "origin_id": 222,
            "origin_slot": 0,
            "target_id": 223,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 408,
            "origin_id": 223,
            "origin_slot": 0,
            "target_id": 176,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 208,
            "origin_id": 114,
            "origin_slot": 0,
            "target_id": 224,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 209,
            "origin_id": 114,
            "origin_slot": 1,
            "target_id": 224,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 403,
            "origin_id": 220,
            "origin_slot": 0,
            "target_id": 235,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 410,
            "origin_id": 111,
            "origin_slot": 0,
            "target_id": 224,
            "target_slot": 3,
            "type": "VAE"
          },
          {
            "id": 411,
            "origin_id": 224,
            "origin_slot": 0,
            "target_id": 225,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 412,
            "origin_id": 224,
            "origin_slot": 1,
            "target_id": 225,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 406,
            "origin_id": 221,
            "origin_slot": 0,
            "target_id": 236,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 413,
            "origin_id": 111,
            "origin_slot": 0,
            "target_id": 225,
            "target_slot": 3,
            "type": "VAE"
          },
          {
            "id": 414,
            "origin_id": 225,
            "origin_slot": 0,
            "target_id": 123,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 415,
            "origin_id": 225,
            "origin_slot": 1,
            "target_id": 123,
            "target_slot": 2,
            "type": "CONDITIONING"
          },
          {
            "id": 417,
            "origin_id": 226,
            "origin_slot": 0,
            "target_id": 228,
            "target_slot": 1,
            "type": "MASK"
          },
          {
            "id": 418,
            "origin_id": 226,
            "origin_slot": 0,
            "target_id": 230,
            "target_slot": 2,
            "type": "MASK"
          },
          {
            "id": 419,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 227,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 420,
            "origin_id": 111,
            "origin_slot": 0,
            "target_id": 227,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 421,
            "origin_id": 227,
            "origin_slot": 0,
            "target_id": 228,
            "target_slot": 0,
            "type": "LATENT"
          },
          {
            "id": 422,
            "origin_id": 228,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 4,
            "type": "LATENT"
          },
          {
            "id": 423,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 230,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 424,
            "origin_id": 131,
            "origin_slot": 0,
            "target_id": 230,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 425,
            "origin_id": 230,
            "origin_slot": 0,
            "target_id": 177,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 501,
            "origin_id": 230,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 502,
            "origin_id": 177,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 1,
            "type": "IMAGE"
          },
          {
            "id": 503,
            "origin_id": 131,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 3,
            "type": "IMAGE"
          },
          {
            "id": 343,
            "origin_id": 197,
            "origin_slot": 0,
            "target_id": 198,
            "target_slot": 0,
            "type": "STRING"
          },
          {
            "id": 426,
            "origin_id": -10,
            "origin_slot": 10,
            "target_id": 231,
            "target_slot": 0,
            "type": "MASK"
          },
          {
            "id": 427,
            "origin_id": -10,
            "origin_slot": 11,
            "target_id": 232,
            "target_slot": 0,
            "type": "MASK"
          },
          {
            "id": 428,
            "origin_id": -10,
            "origin_slot": 12,
            "target_id": 233,
            "target_slot": 0,
            "type": "MASK"
          },
          {
            "id": 429,
            "origin_id": 231,
            "origin_slot": 0,
            "target_id": 234,
            "target_slot": 5,
            "type": "MASK"
          },
          {
            "id": 430,
            "origin_id": 232,
            "origin_slot": 0,
            "target_id": 235,
            "target_slot": 5,
            "type": "MASK"
          },
          {
            "id": 431,
            "origin_id": 233,
            "origin_slot": 0,
            "target_id": 236,
            "target_slot": 5,
            "type": "MASK"
          },
          {
            "id": 432,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 234,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 433,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 235,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 434,
            "origin_id": 120,
            "origin_slot": 0,
            "target_id": 236,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 435,
            "origin_id": 121,
            "origin_slot": 0,
            "target_id": 222,
            "target_slot": 0,
            "type": "IMAGE,MASK"
          },
          {
            "id": 436,
            "origin_id": 220,
            "origin_slot": 0,
            "target_id": 222,
            "target_slot": 1,
            "type": "IMAGE,MASK"
          },
          {
            "id": 437,
            "origin_id": 221,
            "origin_slot": 0,
            "target_id": 223,
            "target_slot": 1,
            "type": "IMAGE,MASK"
          },
          {
            "id": 438,
            "origin_id": 121,
            "origin_slot": 0,
            "target_id": 114,
            "target_slot": 2,
            "type": "IMAGE"
          },
          {
            "id": 439,
            "origin_id": 220,
            "origin_slot": 0,
            "target_id": 224,
            "target_slot": 2,
            "type": "IMAGE"
          },
          {
            "id": 440,
            "origin_id": 221,
            "origin_slot": 0,
            "target_id": 225,
            "target_slot": 2,
            "type": "IMAGE"
          }
        ],
        "name": "Image Generation (Tier 2 High-Speed 4-Step Masked Faceswap)",
        "nodes": [
          {
            "flags": {},
            "id": 110,
            "inputs": [
              {
                "link": 370,
                "localized_name": "noise_seed",
                "name": "noise_seed",
                "type": "INT",
                "widget": {
                  "name": "noise_seed"
                }
              }
            ],
            "mode": 0,
            "order": 5,
            "outputs": [
              {
                "links": [
                  187
                ],
                "localized_name": "NOISE",
                "name": "NOISE",
                "type": "NOISE"
              }
            ],
            "pos": [
              1440,
              3230
            ],
            "properties": {
              "Node name for S&R": "RandomNoise",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              320,
              90
            ],
            "type": "RandomNoise",
            "widgets_values": [
              662064317061748,
              "randomize"
            ]
          },
          {
            "flags": {},
            "id": 111,
            "inputs": [
              {
                "link": 362,
                "localized_name": "vae_name",
                "name": "vae_name",
                "type": "COMBO",
                "widget": {
                  "name": "vae_name"
                }
              }
            ],
            "mode": 0,
            "order": 6,
            "outputs": [
              {
                "links": [
                  198,
                  202,
                  217,
                  410,
                  413,
                  420
                ],
                "localized_name": "VAE",
                "name": "VAE",
                "type": "VAE"
              }
            ],
            "pos": [
              300,
              3610
            ],
            "properties": {
              "Node name for S&R": "VAELoader",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "models": [
                {
                  "directory": "vae",
                  "name": "flux2-vae.safetensors",
                  "url": "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors"
                }
              ],
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              370,
              70
            ],
            "type": "VAELoader",
            "widgets_values": [
              "flux2-vae.safetensors"
            ]
          },
          {
            "flags": {},
            "id": 112,
            "inputs": [
              {
                "link": 351,
                "localized_name": "image",
                "name": "image",
                "type": "IMAGE"
              }
            ],
            "mode": 0,
            "order": 7,
            "outputs": [
              {
                "links": [
                  193,
                  205
                ],
                "localized_name": "width",
                "name": "width",
                "type": "INT"
              },
              {
                "links": [
                  194,
                  206
                ],
                "localized_name": "height",
                "name": "height",
                "type": "INT"
              },
              {
                "links": null,
                "localized_name": "batch_size",
                "name": "batch_size",
                "type": "INT"
              }
            ],
            "pos": [
              1110,
              3840
            ],
            "properties": {
              "Node name for S&R": "GetImageSize",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              230,
              80
            ],
            "type": "GetImageSize",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 114,
            "inputs": [
              {
                "label": "positive",
                "link": 195,
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "link": 196,
                "name": "conditioning_1",
                "type": "CONDITIONING"
              },
              {
                "link": 438,
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 198,
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 9,
            "outputs": [
              {
                "label": "positive",
                "links": [
                  208
                ],
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "links": [
                  209
                ],
                "name": "CONDITIONING_1",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              790,
              4010
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "previewExposures": [],
              "proxyWidgets": [],
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              240,
              120
            ],
            "type": "6e5070f7-26e8-4a9a-ae4d-5d3fc1c591af"
          },
          {
            "flags": {},
            "id": 116,
            "inputs": [
              {
                "label": "positive",
                "link": 199,
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "link": 200,
                "name": "conditioning_1",
                "type": "CONDITIONING"
              },
              {
                "link": 347,
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 202,
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 11,
            "outputs": [
              {
                "label": "positive",
                "links": [
                  195
                ],
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "links": [
                  196
                ],
                "name": "CONDITIONING_1",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              790,
              3840
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "previewExposures": [],
              "proxyWidgets": [],
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              240,
              120
            ],
            "type": "5eb5cfa6-f140-40b7-bc6f-9477d677f888"
          },
          {
            "flags": {},
            "id": 113,
            "inputs": [
              {
                "link": 193,
                "localized_name": "width",
                "name": "width",
                "type": "INT",
                "widget": {
                  "name": "width"
                }
              },
              {
                "link": 194,
                "localized_name": "height",
                "name": "height",
                "type": "INT",
                "widget": {
                  "name": "height"
                }
              },
              {
                "link": null,
                "localized_name": "batch_size",
                "name": "batch_size",
                "type": "INT",
                "widget": {
                  "name": "batch_size"
                }
              }
            ],
            "mode": 0,
            "order": 8,
            "outputs": [
              {
                "links": [
                  191
                ],
                "localized_name": "LATENT",
                "name": "LATENT",
                "type": "LATENT"
              }
            ],
            "pos": [
              1440,
              3840
            ],
            "properties": {
              "Node name for S&R": "EmptyFlux2LatentImage",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              320,
              120
            ],
            "type": "EmptyFlux2LatentImage",
            "widgets_values": [
              1024,
              1024,
              1
            ]
          },
          {
            "flags": {},
            "id": 109,
            "inputs": [
              {
                "link": 187,
                "localized_name": "noise",
                "name": "noise",
                "type": "NOISE"
              },
              {
                "link": 188,
                "localized_name": "guider",
                "name": "guider",
                "type": "GUIDER"
              },
              {
                "link": 189,
                "localized_name": "sampler",
                "name": "sampler",
                "type": "SAMPLER"
              },
              {
                "link": 310,
                "localized_name": "sigmas",
                "name": "sigmas",
                "type": "SIGMAS"
              },
              {
                "link": 422,
                "localized_name": "latent_image",
                "name": "latent_image",
                "type": "LATENT"
              }
            ],
            "mode": 0,
            "order": 4,
            "outputs": [
              {
                "links": [
                  216
                ],
                "localized_name": "output",
                "name": "output",
                "type": "LATENT"
              },
              {
                "links": [],
                "localized_name": "denoised_output",
                "name": "denoised_output",
                "type": "LATENT"
              }
            ],
            "pos": [
              1830,
              3230
            ],
            "properties": {
              "Node name for S&R": "SamplerCustomAdvanced",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              230,
              420
            ],
            "type": "SamplerCustomAdvanced",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 115,
            "inputs": [
              {
                "link": 368,
                "localized_name": "clip_name",
                "name": "clip_name",
                "type": "COMBO",
                "widget": {
                  "name": "clip_name"
                }
              },
              {
                "link": null,
                "localized_name": "type",
                "name": "type",
                "type": "COMBO",
                "widget": {
                  "name": "type"
                }
              },
              {
                "link": null,
                "localized_name": "device",
                "name": "device",
                "shape": 7,
                "type": "COMBO",
                "widget": {
                  "name": "device"
                }
              }
            ],
            "mode": 0,
            "order": 10,
            "outputs": [
              {
                "links": [
                  203,
                  204
                ],
                "localized_name": "CLIP",
                "name": "CLIP",
                "type": "CLIP"
              }
            ],
            "pos": [
              300,
              3410
            ],
            "properties": {
              "Node name for S&R": "CLIPLoader",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "models": [
                {
                  "directory": "text_encoders",
                  "name": "qwen_3_8b_fp8mixed.safetensors",
                  "url": "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
                }
              ],
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              370,
              130
            ],
            "type": "CLIPLoader",
            "widgets_values": [
              "qwen_3_8b_fp8mixed.safetensors",
              "flux2",
              "default"
            ]
          },
          {
            "bgcolor": "#533",
            "color": "#322",
            "flags": {},
            "id": 119,
            "inputs": [
              {
                "link": 204,
                "localized_name": "clip",
                "name": "clip",
                "type": "CLIP"
              },
              {
                "link": null,
                "localized_name": "text",
                "name": "text",
                "type": "STRING",
                "widget": {
                  "name": "text"
                }
              }
            ],
            "mode": 0,
            "order": 14,
            "outputs": [
              {
                "links": [
                  200
                ],
                "localized_name": "CONDITIONING",
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              790,
              3610
            ],
            "properties": {
              "Node name for S&R": "CLIPTextEncode",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.9.1"
            },
            "size": [
              430,
              110
            ],
            "title": "CLIP Text Encode ( Negative Prompt)",
            "type": "CLIPTextEncode",
            "widgets_values": [
              "wide head, bloated face, wide skull, cartoon proportions, stylized head shape, exaggerated cartoon eyes, wide-spaced ears, tiny anime chin, caricature proportions, distorted skull anatomy, doll face, 3d animation geometry, bad quality, noise, blurry, worst quality, low resolution, blur, distortion, unnatural blending, cartoon, illustration, painting, wrong identity, face morphing, mismatched facial features, asymmetrical eyes, double face, seam lines, inconsistent face structure, altered facial identity, white background, solid background, studio backdrop, foreign background bleed, background box, halo artifacts, edge seams"
            ]
          },
          {
            "flags": {},
            "id": 123,
            "inputs": [
              {
                "link": 207,
                "localized_name": "model",
                "name": "model",
                "type": "MODEL"
              },
              {
                "link": 414,
                "localized_name": "positive",
                "name": "positive",
                "type": "CONDITIONING"
              },
              {
                "link": 415,
                "localized_name": "negative",
                "name": "negative",
                "type": "CONDITIONING"
              },
              {
                "link": null,
                "localized_name": "cfg",
                "name": "cfg",
                "type": "FLOAT",
                "widget": {
                  "name": "cfg"
                }
              }
            ],
            "mode": 0,
            "order": 18,
            "outputs": [
              {
                "links": [
                  188
                ],
                "localized_name": "GUIDER",
                "name": "GUIDER",
                "type": "GUIDER"
              }
            ],
            "pos": [
              1440,
              3380
            ],
            "properties": {
              "Node name for S&R": "CFGGuider",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              320,
              110
            ],
            "type": "CFGGuider",
            "widgets_values": [
              1.0
            ],
            "title": "Distilled CFG Guider (CFG 1.0)"
          },
          {
            "flags": {},
            "id": 176,
            "inputs": [
              {
                "link": 353,
                "localized_name": "image1",
                "name": "image1",
                "type": "IMAGE,MASK"
              },
              {
                "link": 408,
                "localized_name": "image2",
                "name": "image2",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "direction",
                "name": "direction",
                "type": "COMBO",
                "widget": {
                  "name": "direction"
                }
              },
              {
                "link": null,
                "localized_name": "match_image_size",
                "name": "match_image_size",
                "type": "BOOLEAN",
                "widget": {
                  "name": "match_image_size"
                }
              }
            ],
            "mode": 0,
            "order": 22,
            "outputs": [
              {
                "links": [
                  304
                ],
                "localized_name": "output",
                "name": "output",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              1850,
              3840
            ],
            "properties": {
              "Node name for S&R": "ImageConcanate",
              "cnr_id": "comfyui-kjnodes",
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "93a4fba44c1635a716dc262f157a7ded7d716f3a"
            },
            "size": [
              270,
              140
            ],
            "type": "ImageConcanate",
            "widgets_values": [
              "down",
              true
            ]
          },
          {
            "flags": {},
            "id": 131,
            "inputs": [
              {
                "link": 216,
                "localized_name": "samples",
                "name": "samples",
                "type": "LATENT"
              },
              {
                "link": 217,
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 19,
            "outputs": [
              {
                "links": [
                  424,
                  503
                ],
                "localized_name": "IMAGE",
                "name": "IMAGE",
                "slot_index": 0,
                "type": "IMAGE"
              }
            ],
            "pos": [
              2100,
              3230
            ],
            "properties": {
              "Node name for S&R": "VAEDecode",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              230,
              60
            ],
            "type": "VAEDecode",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 177,
            "inputs": [
              {
                "link": 304,
                "localized_name": "image1",
                "name": "image1",
                "type": "IMAGE,MASK"
              },
              {
                "link": 425,
                "localized_name": "image2",
                "name": "image2",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "direction",
                "name": "direction",
                "type": "COMBO",
                "widget": {
                  "name": "direction"
                }
              },
              {
                "link": null,
                "localized_name": "match_image_size",
                "name": "match_image_size",
                "type": "BOOLEAN",
                "widget": {
                  "name": "match_image_size"
                }
              }
            ],
            "mode": 0,
            "order": 23,
            "outputs": [
              {
                "links": [
                  306
                ],
                "localized_name": "output",
                "name": "output",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              2190,
              3840
            ],
            "properties": {
              "Node name for S&R": "ImageConcanate",
              "cnr_id": "comfyui-kjnodes",
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "93a4fba44c1635a716dc262f157a7ded7d716f3a"
            },
            "size": [
              270,
              140
            ],
            "type": "ImageConcanate",
            "widgets_values": [
              "right",
              true
            ]
          },
          {
            "flags": {},
            "id": 120,
            "inputs": [
              {
                "link": 221,
                "localized_name": "input",
                "name": "input",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "resize_type",
                "name": "resize_type",
                "type": "COMFY_DYNAMICCOMBO_V3",
                "widget": {
                  "name": "resize_type"
                }
              },
              {
                "link": null,
                "localized_name": "resize_type.megapixels",
                "name": "resize_type.megapixels",
                "type": "FLOAT",
                "widget": {
                  "name": "resize_type.megapixels"
                }
              },
              {
                "link": null,
                "localized_name": "scale_method",
                "name": "scale_method",
                "type": "COMBO",
                "widget": {
                  "name": "scale_method"
                }
              }
            ],
            "mode": 0,
            "order": 15,
            "outputs": [
              {
                "links": [
                  335,
                  347,
                  351,
                  352,
                  353,
                  419,
                  423
                ],
                "localized_name": "resized",
                "name": "resized",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              -410,
              3030
            ],
            "properties": {
              "Node name for S&R": "ResizeImageMaskNode",
              "cnr_id": "comfy-core",
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.9.2"
            },
            "size": [
              370,
              110
            ],
            "type": "ResizeImageMaskNode",
            "widgets_values": [
              "scale total pixels",
              1,
              "lanczos"
            ]
          },
          {
            "flags": {},
            "id": 121,
            "inputs": [
              {
                "link": 282,
                "localized_name": "input",
                "name": "input",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "resize_type",
                "name": "resize_type",
                "type": "COMFY_DYNAMICCOMBO_V3",
                "widget": {
                  "name": "resize_type"
                }
              },
              {
                "link": null,
                "localized_name": "resize_type.megapixels",
                "name": "resize_type.megapixels",
                "type": "FLOAT",
                "widget": {
                  "name": "resize_type.megapixels"
                }
              },
              {
                "link": null,
                "localized_name": "scale_method",
                "name": "scale_method",
                "type": "COMBO",
                "widget": {
                  "name": "scale_method"
                }
              }
            ],
            "mode": 0,
            "order": 16,
            "outputs": [
              {
                "links": [
                  349,
                  350
                ],
                "localized_name": "resized",
                "name": "resized",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              -410,
              3200
            ],
            "properties": {
              "Node name for S&R": "ResizeImageMaskNode",
              "cnr_id": "comfy-core",
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.9.2"
            },
            "size": [
              370,
              110
            ],
            "type": "ResizeImageMaskNode",
            "widgets_values": [
              "scale total pixels",
              1,
              "lanczos"
            ]
          },
          {
            "flags": {},
            "id": 118,
            "inputs": [
              {
                "link": 369,
                "localized_name": "unet_name",
                "name": "unet_name",
                "type": "COMBO",
                "widget": {
                  "name": "unet_name"
                }
              },
              {
                "link": null,
                "localized_name": "weight_dtype",
                "name": "weight_dtype",
                "type": "COMBO",
                "widget": {
                  "name": "weight_dtype"
                }
              }
            ],
            "mode": 0,
            "order": 13,
            "outputs": [
              {
                "links": [
                  317
                ],
                "localized_name": "MODEL",
                "name": "MODEL",
                "type": "MODEL"
              }
            ],
            "pos": [
              300,
              3240
            ],
            "properties": {
              "Node name for S&R": "UNETLoader",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2",
              "models": [
                {
                  "directory": "diffusion_models",
                  "name": "flux-2-klein-9b.safetensors",
                  "url": "https://huggingface.co/MIUProject/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors"
                }
              ]
            },
            "size": [
              370,
              100
            ],
            "type": "UNETLoader",
            "widgets_values": [
              "flux-2-klein-9b.safetensors",
              "default"
            ]
          },
          {
            "flags": {},
            "id": 122,
            "inputs": [
              {
                "link": null,
                "localized_name": "steps",
                "name": "steps",
                "type": "INT",
                "widget": {
                  "name": "steps"
                }
              },
              {
                "link": 205,
                "localized_name": "width",
                "name": "width",
                "type": "INT",
                "widget": {
                  "name": "width"
                }
              },
              {
                "link": 206,
                "localized_name": "height",
                "name": "height",
                "type": "INT",
                "widget": {
                  "name": "height"
                }
              }
            ],
            "mode": 0,
            "order": 17,
            "outputs": [
              {
                "links": [
                  310
                ],
                "localized_name": "SIGMAS",
                "name": "SIGMAS",
                "type": "SIGMAS"
              }
            ],
            "pos": [
              1440,
              3670
            ],
            "properties": {
              "Node name for S&R": "Flux2Scheduler",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              320,
              120
            ],
            "type": "Flux2Scheduler",
            "widgets_values": [
              8,
              1024,
              1024
            ],
            "title": "8-Step Morphology-Accurate Flux Scheduler"
          },
          {
            "flags": {},
            "id": 125,
            "inputs": [
              {
                "link": null,
                "localized_name": "sampler_name",
                "name": "sampler_name",
                "type": "COMBO",
                "widget": {
                  "name": "sampler_name"
                }
              }
            ],
            "mode": 0,
            "order": 0,
            "outputs": [
              {
                "links": [
                  189
                ],
                "localized_name": "SAMPLER",
                "name": "SAMPLER",
                "type": "SAMPLER"
              }
            ],
            "pos": [
              1440,
              3540
            ],
            "properties": {
              "Node name for S&R": "KSamplerSelect",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              320,
              70
            ],
            "type": "KSamplerSelect",
            "widgets_values": [
              "euler"
            ]
          },
          {
            "bgcolor": "#353",
            "color": "#232",
            "flags": {},
            "id": 117,
            "inputs": [
              {
                "link": 203,
                "localized_name": "clip",
                "name": "clip",
                "type": "CLIP"
              },
              {
                "link": 344,
                "localized_name": "text",
                "name": "text",
                "type": "STRING",
                "widget": {
                  "name": "text"
                }
              }
            ],
            "mode": 0,
            "order": 12,
            "outputs": [
              {
                "links": [
                  199
                ],
                "localized_name": "CONDITIONING",
                "name": "CONDITIONING",
                "slot_index": 0,
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              790,
              3230
            ],
            "properties": {
              "Node name for S&R": "CLIPTextEncode",
              "cnr_id": "comfy-core",
              "enableTabs": false,
              "hasSecondTab": false,
              "secondTabOffset": 80,
              "secondTabText": "Send Back",
              "secondTabWidth": 65,
              "tabWidth": 65,
              "tabXOffset": 10,
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.8.2"
            },
            "size": [
              440,
              320
            ],
            "title": "CLIP Text Encode (Positive Prompt)",
            "type": "CLIPTextEncode",
            "widgets_values": [
              ""
            ]
          },
          {
            "flags": {},
            "id": 133,
            "inputs": [
              {
                "link": 317,
                "localized_name": "model",
                "name": "model",
                "type": "MODEL"
              },
              {
                "link": 363,
                "localized_name": "lora_name",
                "name": "lora_name",
                "type": "COMBO",
                "widget": {
                  "name": "lora_name"
                }
              },
              {
                "link": null,
                "localized_name": "strength_model",
                "name": "strength_model",
                "type": "FLOAT",
                "widget": {
                  "name": "strength_model"
                }
              }
            ],
            "mode": 0,
            "order": 20,
            "outputs": [
              {
                "links": [
                  207
                ],
                "localized_name": "MODEL",
                "name": "MODEL",
                "type": "MODEL"
              }
            ],
            "pos": [
              800,
              3040
            ],
            "properties": {
              "Node name for S&R": "LoraLoaderModelOnly",
              "cnr_id": "comfy-core",
              "ue_properties": {
                "input_ue_unconnectable": {},
                "version": "7.5.1",
                "widget_ue_connectable": {}
              },
              "ver": "0.9.2",
              "models": [
                {
                  "directory": "loras",
                  "name": "Alissonerdx__BFS-Best-Face-Swap__bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
                  "url": "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors"
                }
              ]
            },
            "size": [
              440,
              100
            ],
            "type": "LoraLoaderModelOnly",
            "widgets_values": [
              "Alissonerdx__BFS-Best-Face-Swap__bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
              1.22
            ],
            "title": "LoRA Loader (BFS 1.22x Strong Identity Dominance)"
          },
          {
            "flags": {},
            "id": 197,
            "inputs": [
              {
                "link": null,
                "localized_name": "value",
                "name": "value",
                "type": "STRING",
                "widget": {
                  "name": "value"
                }
              }
            ],
            "mode": 0,
            "order": 3,
            "outputs": [
              {
                "links": [
                  343
                ],
                "localized_name": "STRING",
                "name": "STRING",
                "type": "STRING"
              }
            ],
            "pos": [
              980,
              2710
            ],
            "properties": {
              "Node name for S&R": "PrimitiveStringMultiline"
            },
            "size": [
              330,
              200
            ],
            "title": "Prompt Base",
            "type": "PrimitiveStringMultiline",
            "widgets_values": [
              "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, clothing, and background. In Picture 1, replace the masked face and entire head with the exact biological facial identity, natural skull shape, and true proportions of the child shown in Picture 2, Picture 3, and Picture 4 (multi-angle reference photos). Extract facial features, hair, identity, and head shape only; completely ignore any foreign or solid backgrounds present in reference photos. \n\nCRITICAL MORPHOLOGY & HEAD WIDTH OVERRIDE: Completely override and reconstruct the skull shape, head width, jawline, chin, and cheekbones to match the realistic human child in Picture 2, Picture 3, and Picture 4. Do NOT inherit cartoon proportions, wide anime head silhouette, exaggerated giant eyes, wide-spaced ears, or swollen cheeks from Picture 1. Strictly enforce natural, realistic child head width, slender natural cheeks, realistic narrow eye-to-temple spacing, and an anatomically proportionate child head contour. Match the head tilt, eye gaze direction, ambient lighting, skin tone, shadows, and micro-expressions with photorealistic precision, seamless edge blending, razor sharp facial details, 8k resolution, ultra-detailed natural skin texture.\n"
            ]
          },
          {
            "flags": {},
            "id": 198,
            "inputs": [
              {
                "link": 343,
                "localized_name": "source",
                "name": "source",
                "type": "*"
              }
            ],
            "mode": 0,
            "order": 26,
            "outputs": [
              {
                "links": [
                  344
                ],
                "localized_name": "STRING",
                "name": "STRING",
                "type": "STRING"
              }
            ],
            "pos": [
              1690,
              2450
            ],
            "properties": {
              "Node name for S&R": "PreviewAny"
            },
            "size": [
              430,
              460
            ],
            "type": "PreviewAny",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 220,
            "inputs": [
              {
                "link": 401,
                "localized_name": "input",
                "name": "input",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "resize_type",
                "name": "resize_type",
                "type": "COMFY_DYNAMICCOMBO_V3",
                "widget": {
                  "name": "resize_type"
                }
              },
              {
                "link": null,
                "localized_name": "resize_type.megapixels",
                "name": "resize_type.megapixels",
                "type": "FLOAT",
                "widget": {
                  "name": "resize_type.megapixels"
                }
              },
              {
                "link": null,
                "localized_name": "scale_method",
                "name": "scale_method",
                "type": "COMBO",
                "widget": {
                  "name": "scale_method"
                }
              }
            ],
            "mode": 0,
            "order": 16,
            "outputs": [
              {
                "links": [
                  402,
                  403
                ],
                "localized_name": "resized",
                "name": "resized",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              -410,
              3370
            ],
            "properties": {
              "Node name for S&R": "ResizeImageMaskNode",
              "cnr_id": "comfy-core",
              "ver": "0.9.2"
            },
            "size": [
              370,
              110
            ],
            "title": "Resize Kid A Ref 2",
            "type": "ResizeImageMaskNode",
            "widgets_values": [
              "scale total pixels",
              1,
              "lanczos"
            ]
          },
          {
            "flags": {},
            "id": 221,
            "inputs": [
              {
                "link": 404,
                "localized_name": "input",
                "name": "input",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "resize_type",
                "name": "resize_type",
                "type": "COMFY_DYNAMICCOMBO_V3",
                "widget": {
                  "name": "resize_type"
                }
              },
              {
                "link": null,
                "localized_name": "resize_type.megapixels",
                "name": "resize_type.megapixels",
                "type": "FLOAT",
                "widget": {
                  "name": "resize_type.megapixels"
                }
              },
              {
                "link": null,
                "localized_name": "scale_method",
                "name": "scale_method",
                "type": "COMBO",
                "widget": {
                  "name": "scale_method"
                }
              }
            ],
            "mode": 0,
            "order": 17,
            "outputs": [
              {
                "links": [
                  405,
                  406
                ],
                "localized_name": "resized",
                "name": "resized",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              -410,
              3540
            ],
            "properties": {
              "Node name for S&R": "ResizeImageMaskNode",
              "cnr_id": "comfy-core",
              "ver": "0.9.2"
            },
            "size": [
              370,
              110
            ],
            "title": "Resize Kid A Ref 3",
            "type": "ResizeImageMaskNode",
            "widgets_values": [
              "scale total pixels",
              1,
              "lanczos"
            ]
          },
          {
            "flags": {},
            "id": 222,
            "inputs": [
              {
                "link": 435,
                "localized_name": "image1",
                "name": "image1",
                "type": "IMAGE,MASK"
              },
              {
                "link": 436,
                "localized_name": "image2",
                "name": "image2",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "direction",
                "name": "direction",
                "type": "COMBO",
                "widget": {
                  "name": "direction"
                }
              },
              {
                "link": null,
                "localized_name": "match_image_size",
                "name": "match_image_size",
                "type": "BOOLEAN",
                "widget": {
                  "name": "match_image_size"
                }
              }
            ],
            "mode": 0,
            "order": 18,
            "outputs": [
              {
                "links": [
                  407
                ],
                "localized_name": "output",
                "name": "output",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              0,
              3350
            ],
            "properties": {
              "Node name for S&R": "ImageConcanate",
              "cnr_id": "comfyui-kjnodes",
              "ver": "93a4fba"
            },
            "size": [
              270,
              140
            ],
            "title": "Combine Ref 1 & Ref 2",
            "type": "ImageConcanate",
            "widgets_values": [
              "right",
              true
            ]
          },
          {
            "flags": {},
            "id": 223,
            "inputs": [
              {
                "link": 407,
                "localized_name": "image1",
                "name": "image1",
                "type": "IMAGE,MASK"
              },
              {
                "link": 437,
                "localized_name": "image2",
                "name": "image2",
                "type": "IMAGE,MASK"
              },
              {
                "link": null,
                "localized_name": "direction",
                "name": "direction",
                "type": "COMBO",
                "widget": {
                  "name": "direction"
                }
              },
              {
                "link": null,
                "localized_name": "match_image_size",
                "name": "match_image_size",
                "type": "BOOLEAN",
                "widget": {
                  "name": "match_image_size"
                }
              }
            ],
            "mode": 0,
            "order": 19,
            "outputs": [
              {
                "links": [
                  408
                ],
                "localized_name": "output",
                "name": "output",
                "type": "IMAGE,MASK"
              }
            ],
            "pos": [
              300,
              3350
            ],
            "properties": {
              "Node name for S&R": "ImageConcanate",
              "cnr_id": "comfyui-kjnodes",
              "ver": "93a4fba"
            },
            "size": [
              270,
              140
            ],
            "title": "Combine All 3 Kid A Refs",
            "type": "ImageConcanate",
            "widgets_values": [
              "right",
              true
            ]
          },
          {
            "flags": {},
            "id": 224,
            "inputs": [
              {
                "label": "positive",
                "link": 208,
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "link": 209,
                "name": "conditioning_1",
                "type": "CONDITIONING"
              },
              {
                "link": 439,
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 410,
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 20,
            "outputs": [
              {
                "label": "positive",
                "links": [
                  411
                ],
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "links": [
                  412
                ],
                "name": "CONDITIONING_1",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              790,
              4180
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.8.2"
            },
            "size": [
              240,
              120
            ],
            "title": "Reference Conditioning (Kid A Ref 2)",
            "type": "6e5070f7-26e8-4a9a-ae4d-5d3fc1c591af"
          },
          {
            "flags": {},
            "id": 225,
            "inputs": [
              {
                "label": "positive",
                "link": 411,
                "name": "conditioning",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "link": 412,
                "name": "conditioning_1",
                "type": "CONDITIONING"
              },
              {
                "link": 440,
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 413,
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 21,
            "outputs": [
              {
                "label": "positive",
                "links": [
                  414
                ],
                "name": "CONDITIONING",
                "type": "CONDITIONING"
              },
              {
                "label": "negative",
                "links": [
                  415
                ],
                "name": "CONDITIONING_1",
                "type": "CONDITIONING"
              }
            ],
            "pos": [
              790,
              4350
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.8.2"
            },
            "size": [
              240,
              120
            ],
            "title": "Reference Conditioning (Kid A Ref 3)",
            "type": "6e5070f7-26e8-4a9a-ae4d-5d3fc1c591af"
          },
          {
            "flags": {},
            "id": 226,
            "inputs": [
              {
                "link": 416,
                "localized_name": "mask",
                "name": "mask",
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 22,
            "outputs": [
              {
                "links": [
                  417,
                  418
                ],
                "localized_name": "MASK",
                "name": "MASK",
                "type": "MASK"
              }
            ],
            "pos": [
              450,
              2900
            ],
            "properties": {
              "Node name for S&R": "GrowMask",
              "cnr_id": "comfy-core",
              "ver": "0.24.0"
            },
            "size": [
              210,
              120
            ],
            "title": "GrowMask (Expanded Coverage for Skull & Jaw Reshaping)",
            "type": "GrowMask",
            "widgets_values": [
              52,
              true
            ]
          },
          {
            "flags": {},
            "id": 227,
            "inputs": [
              {
                "link": 419,
                "localized_name": "pixels",
                "name": "pixels",
                "type": "IMAGE"
              },
              {
                "link": 420,
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE"
              }
            ],
            "mode": 0,
            "order": 23,
            "outputs": [
              {
                "links": [
                  421
                ],
                "localized_name": "LATENT",
                "name": "LATENT",
                "type": "LATENT"
              }
            ],
            "pos": [
              750,
              2900
            ],
            "properties": {
              "Node name for S&R": "VAEEncode",
              "cnr_id": "comfy-core",
              "ver": "0.8.2"
            },
            "size": [
              220,
              100
            ],
            "title": "Target Scene VAE Encode",
            "type": "VAEEncode",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 228,
            "inputs": [
              {
                "link": 421,
                "localized_name": "samples",
                "name": "samples",
                "type": "LATENT"
              },
              {
                "link": 417,
                "localized_name": "mask",
                "name": "mask",
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 24,
            "outputs": [
              {
                "links": [
                  422
                ],
                "localized_name": "LATENT",
                "name": "LATENT",
                "type": "LATENT"
              }
            ],
            "pos": [
              1050,
              2900
            ],
            "properties": {
              "Node name for S&R": "SetLatentNoiseMask",
              "cnr_id": "comfy-core",
              "ver": "0.24.0"
            },
            "size": [
              210,
              100
            ],
            "title": "Set Face Noise Mask",
            "type": "SetLatentNoiseMask",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 230,
            "inputs": [
              {
                "link": 423,
                "localized_name": "destination",
                "name": "destination",
                "type": "IMAGE"
              },
              {
                "link": 424,
                "localized_name": "source",
                "name": "source",
                "type": "IMAGE"
              },
              {
                "link": 418,
                "localized_name": "mask",
                "name": "mask",
                "type": "MASK",
                "shape": 7
              }
            ],
            "mode": 0,
            "order": 27,
            "outputs": [
              {
                "links": [
                  501,
                  425
                ],
                "localized_name": "IMAGE",
                "name": "IMAGE",
                "type": "IMAGE"
              }
            ],
            "pos": [
              2380,
              3230
            ],
            "properties": {
              "Node name for S&R": "ImageCompositeMasked",
              "cnr_id": "comfy-core",
              "ver": "0.24.0"
            },
            "size": [
              230,
              240
            ],
            "title": "Pristine Background Composite",
            "type": "ImageCompositeMasked",
            "widgets_values": [
              0,
              0,
              false
            ]
          },
          {
            "flags": {},
            "id": 231,
            "inputs": [
              {
                "link": 426,
                "localized_name": "mask",
                "name": "mask",
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 1,
            "outputs": [
              {
                "links": [
                  429
                ],
                "localized_name": "MASK",
                "name": "MASK",
                "type": "MASK"
              }
            ],
            "pos": [
              -200,
              3475
            ],
            "properties": {
              "Node name for S&R": "InvertMask",
              "cnr_id": "comfy-core"
            },
            "size": [
              210,
              30
            ],
            "type": "InvertMask",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 232,
            "inputs": [
              {
                "link": 427,
                "localized_name": "mask",
                "name": "mask",
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 1,
            "outputs": [
              {
                "links": [
                  430
                ],
                "localized_name": "MASK",
                "name": "MASK",
                "type": "MASK"
              }
            ],
            "pos": [
              -200,
              3495
            ],
            "properties": {
              "Node name for S&R": "InvertMask",
              "cnr_id": "comfy-core"
            },
            "size": [
              210,
              30
            ],
            "type": "InvertMask",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 233,
            "inputs": [
              {
                "link": 428,
                "localized_name": "mask",
                "name": "mask",
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 1,
            "outputs": [
              {
                "links": [
                  431
                ],
                "localized_name": "MASK",
                "name": "MASK",
                "type": "MASK"
              }
            ],
            "pos": [
              -200,
              3515
            ],
            "properties": {
              "Node name for S&R": "InvertMask",
              "cnr_id": "comfy-core"
            },
            "size": [
              210,
              30
            ],
            "type": "InvertMask",
            "widgets_values": []
          },
          {
            "flags": {},
            "id": 234,
            "inputs": [
              {
                "link": 432,
                "localized_name": "destination",
                "name": "destination",
                "type": "IMAGE"
              },
              {
                "link": 349,
                "localized_name": "source",
                "name": "source",
                "type": "IMAGE"
              },
              {
                "link": null,
                "localized_name": "x",
                "name": "x",
                "type": "INT",
                "widget": {
                  "name": "x"
                }
              },
              {
                "link": null,
                "localized_name": "y",
                "name": "y",
                "type": "INT",
                "widget": {
                  "name": "y"
                }
              },
              {
                "link": null,
                "localized_name": "resize_source",
                "name": "resize_source",
                "type": "BOOLEAN",
                "widget": {
                  "name": "resize_source"
                }
              },
              {
                "link": 429,
                "localized_name": "mask",
                "name": "mask",
                "shape": 7,
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 2,
            "outputs": [
              {
                "links": [
                  435,
                  438
                ],
                "localized_name": "IMAGE",
                "name": "IMAGE",
                "type": "IMAGE"
              }
            ],
            "pos": [
              100,
              3475
            ],
            "properties": {
              "Node name for S&R": "ImageCompositeMasked",
              "cnr_id": "comfy-core"
            },
            "size": [
              280,
              140
            ],
            "type": "ImageCompositeMasked",
            "widgets_values": [
              0,
              0,
              true
            ]
          },
          {
            "flags": {},
            "id": 235,
            "inputs": [
              {
                "link": 433,
                "localized_name": "destination",
                "name": "destination",
                "type": "IMAGE"
              },
              {
                "link": 403,
                "localized_name": "source",
                "name": "source",
                "type": "IMAGE"
              },
              {
                "link": null,
                "localized_name": "x",
                "name": "x",
                "type": "INT",
                "widget": {
                  "name": "x"
                }
              },
              {
                "link": null,
                "localized_name": "y",
                "name": "y",
                "type": "INT",
                "widget": {
                  "name": "y"
                }
              },
              {
                "link": null,
                "localized_name": "resize_source",
                "name": "resize_source",
                "type": "BOOLEAN",
                "widget": {
                  "name": "resize_source"
                }
              },
              {
                "link": 430,
                "localized_name": "mask",
                "name": "mask",
                "shape": 7,
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 2,
            "outputs": [
              {
                "links": [
                  436,
                  439
                ],
                "localized_name": "IMAGE",
                "name": "IMAGE",
                "type": "IMAGE"
              }
            ],
            "pos": [
              100,
              3630
            ],
            "properties": {
              "Node name for S&R": "ImageCompositeMasked",
              "cnr_id": "comfy-core"
            },
            "size": [
              280,
              140
            ],
            "type": "ImageCompositeMasked",
            "widgets_values": [
              0,
              0,
              true
            ]
          },
          {
            "flags": {},
            "id": 236,
            "inputs": [
              {
                "link": 434,
                "localized_name": "destination",
                "name": "destination",
                "type": "IMAGE"
              },
              {
                "link": 406,
                "localized_name": "source",
                "name": "source",
                "type": "IMAGE"
              },
              {
                "link": null,
                "localized_name": "x",
                "name": "x",
                "type": "INT",
                "widget": {
                  "name": "x"
                }
              },
              {
                "link": null,
                "localized_name": "y",
                "name": "y",
                "type": "INT",
                "widget": {
                  "name": "y"
                }
              },
              {
                "link": null,
                "localized_name": "resize_source",
                "name": "resize_source",
                "type": "BOOLEAN",
                "widget": {
                  "name": "resize_source"
                }
              },
              {
                "link": 431,
                "localized_name": "mask",
                "name": "mask",
                "shape": 7,
                "type": "MASK"
              }
            ],
            "mode": 0,
            "order": 2,
            "outputs": [
              {
                "links": [
                  437,
                  440
                ],
                "localized_name": "IMAGE",
                "name": "IMAGE",
                "type": "IMAGE"
              }
            ],
            "pos": [
              100,
              3785
            ],
            "properties": {
              "Node name for S&R": "ImageCompositeMasked",
              "cnr_id": "comfy-core"
            },
            "size": [
              280,
              140
            ],
            "type": "ImageCompositeMasked",
            "widgets_values": [
              0,
              0,
              true
            ]
          }
        ],
        "outputNode": {
          "bounding": [
            2620,
            3201,
            140,
            130
          ],
          "id": -20
        },
        "outputs": [
          {
            "id": "0d6c8071-b99e-461c-a8f8-b80afc0a9fa3",
            "name": "IMAGE",
            "localized_name": "composited",
            "type": "IMAGE",
            "linkIds": [
              501
            ],
            "pos": [
              2640,
              3225
            ]
          },
          {
            "id": "293519ba-f57e-4901-af1c-6271b2878290",
            "name": "output",
            "localized_name": "stitched",
            "type": "IMAGE",
            "linkIds": [
              502
            ],
            "pos": [
              2640,
              3245
            ]
          },
          {
            "id": "ef0f7c92-46a1-4329-8f59-d7f3512cf538",
            "name": "resized",
            "localized_name": "target_orig",
            "type": "IMAGE",
            "linkIds": [
              352
            ],
            "pos": [
              2640,
              3265
            ]
          },
          {
            "id": "f8e7d6c5-0004-461c-a8f8-b80afc0a9fa3",
            "name": "raw_face",
            "localized_name": "raw_face",
            "type": "IMAGE",
            "linkIds": [
              503
            ],
            "pos": [
              2640,
              3285
            ]
          }
        ],
        "reroutes": [
          {
            "id": 2,
            "linkIds": [
              349,
              350
            ],
            "pos": [
              370,
              4074
            ]
          },
          {
            "id": 4,
            "linkIds": [
              347,
              351,
              352,
              353
            ],
            "pos": [
              360,
              3874
            ]
          }
        ],
        "revision": 0,
        "state": {
          "lastGroupId": 7,
          "lastLinkId": 379,
          "lastNodeId": 206,
          "lastRerouteId": 4
        },
        "version": 1,
        "widgets": []
      }
    ]
  },
  "extra": {
    "VHS_KeepIntermediate": true,
    "VHS_MetadataImage": true,
    "VHS_latentpreview": true,
    "VHS_latentpreviewrate": 0,
    "ds": {
      "offset": [
        1035.4650132119818,
        -2789.394757451012
      ],
      "scale": 0.7697274817796108
    },
    "frontendVersion": "1.36.14",
    "linearData": {
      "inputs": [
        [
          "92112d97-bb64-4b44-86f2-ea5691ef8f6e:76:image",
          "image"
        ],
        [
          "92112d97-bb64-4b44-86f2-ea5691ef8f6e:81:image",
          "image"
        ],
        [
          "92112d97-bb64-4b44-86f2-ea5691ef8f6e:82:image",
          "image"
        ],
        [
          "92112d97-bb64-4b44-86f2-ea5691ef8f6e:83:image",
          "image"
        ]
      ],
      "outputs": [
        "204",
        "205",
        "206"
      ]
    },
    "linearMode": true,
    "links_added_by_ue": [],
    "ue_links": [],
    "workflowRendererVersion": "LG"
  },
  "groups": [
    {
      "bounding": [
        -1240,
        2600,
        380,
        2280
      ],
      "color": "#1565c0",
      "flags": {},
      "id": 1,
      "title": "Input Images (Kid B Scene + Kid A 3-Refs)"
    },
    {
      "bounding": [
        -730,
        2950,
        420,
        500
      ],
      "color": "#6a1b9a",
      "flags": {},
      "id": 2,
      "title": "FLUX.2 Tier 2 High-Speed 4-Step 3-Ref Masked Faceswap Engine"
    },
    {
      "bounding": [
        -230,
        2800,
        500,
        1420
      ],
      "color": "#2e7d32",
      "flags": {},
      "id": 3,
      "title": "Image Outputs (Composite, Stitched, Raw)"
    },
    {
      "bounding": [
        270,
        2800,
        500,
        580
      ],
      "color": "#e65100",
      "flags": {},
      "id": 4,
      "title": "Interactive Inspection Slider"
    }
  ],
  "id": "92112d97-bb64-4b44-86f2-ea5691ef8f6e",
  "last_link_id": 607,
  "last_node_id": 250,
  "links": [
    [
      354,
      76,
      0,
      199,
      5,
      "IMAGE"
    ],
    [
      601,
      76,
      1,
      199,
      6,
      "MASK"
    ],
    [
      355,
      81,
      0,
      199,
      7,
      "IMAGE"
    ],
    [
      602,
      82,
      0,
      199,
      8,
      "IMAGE"
    ],
    [
      603,
      83,
      0,
      199,
      9,
      "IMAGE"
    ],
    [
      377,
      199,
      0,
      204,
      0,
      "IMAGE"
    ],
    [
      378,
      199,
      1,
      205,
      0,
      "IMAGE"
    ],
    [
      358,
      199,
      2,
      190,
      1,
      "IMAGE"
    ],
    [
      604,
      199,
      3,
      206,
      0,
      "IMAGE"
    ],
    [
      379,
      204,
      0,
      190,
      0,
      "IMAGE"
    ],
    [
      605,
      81,
      1,
      199,
      10,
      "MASK"
    ],
    [
      606,
      82,
      1,
      199,
      11,
      "MASK"
    ],
    [
      607,
      83,
      1,
      199,
      12,
      "MASK"
    ]
  ],
  "nodes": [
    {
      "flags": {},
      "id": 81,
      "inputs": [
        {
          "label": "Load Source Face",
          "link": null,
          "localized_name": "image",
          "name": "image",
          "type": "COMBO",
          "widget": {
            "name": "image"
          }
        },
        {
          "link": null,
          "localized_name": "choose file to upload",
          "name": "upload",
          "type": "IMAGEUPLOAD",
          "widget": {
            "name": "upload"
          }
        }
      ],
      "mode": 0,
      "order": 0,
      "outputs": [
        {
          "links": [
            355
          ],
          "localized_name": "IMAGE",
          "name": "IMAGE",
          "type": "IMAGE"
        },
        {
          "links": [],
          "localized_name": "MASK",
          "name": "MASK",
          "type": "MASK"
        }
      ],
      "pos": [
        -1200,
        3250
      ],
      "properties": {
        "Node name for S&R": "LoadImage",
        "cnr_id": "comfy-core",
        "enableTabs": false,
        "hasSecondTab": false,
        "image": "clipspace/clipspace-painted-masked-1768744191181.png [input]",
        "secondTabOffset": 80,
        "secondTabText": "Send Back",
        "secondTabWidth": 65,
        "tabWidth": 65,
        "tabXOffset": 10,
        "ue_properties": {
          "input_ue_unconnectable": {},
          "version": "7.5.1",
          "widget_ue_connectable": {}
        },
        "ver": "0.8.2"
      },
      "size": [
        320,
        480
      ],
      "title": "Kid A Reference 1 (Frontal Portrait)",
      "type": "LoadImage",
      "widgets_values": [
        "child_1.png",
        "image"
      ]
    },
    {
      "flags": {},
      "id": 76,
      "inputs": [
        {
          "label": "Load Target Image",
          "link": null,
          "localized_name": "image",
          "name": "image",
          "type": "COMBO",
          "widget": {
            "name": "image"
          }
        },
        {
          "link": null,
          "localized_name": "choose file to upload",
          "name": "upload",
          "type": "IMAGEUPLOAD",
          "widget": {
            "name": "upload"
          }
        }
      ],
      "mode": 0,
      "order": 1,
      "outputs": [
        {
          "links": [
            354
          ],
          "localized_name": "IMAGE",
          "name": "IMAGE",
          "type": "IMAGE"
        },
        {
          "links": [
            601
          ],
          "localized_name": "MASK",
          "name": "MASK",
          "type": "MASK"
        }
      ],
      "pos": [
        -1200,
        2700
      ],
      "properties": {
        "Node name for S&R": "LoadImage",
        "cnr_id": "comfy-core",
        "enableTabs": false,
        "hasSecondTab": false,
        "secondTabOffset": 80,
        "secondTabText": "Send Back",
        "secondTabWidth": 65,
        "tabWidth": 65,
        "tabXOffset": 10,
        "ue_properties": {
          "input_ue_unconnectable": {},
          "version": "7.5.1",
          "widget_ue_connectable": {}
        },
        "ver": "0.8.2"
      },
      "size": [
        320,
        470
      ],
      "title": "Target Scene (Kid B with Face to Mask)",
      "type": "LoadImage",
      "widgets_values": [
        "7b6785c1-e3da-409d-97d8-178a5deaee19.jpg",
        "image"
      ]
    },
    {
      "flags": {},
      "id": 190,
      "inputs": [
        {
          "link": 379,
          "localized_name": "image_a",
          "name": "image_a",
          "shape": 7,
          "type": "IMAGE"
        },
        {
          "link": 358,
          "localized_name": "image_b",
          "name": "image_b",
          "shape": 7,
          "type": "IMAGE"
        },
        {
          "link": null,
          "localized_name": "compare_view",
          "name": "compare_view",
          "type": "IMAGECOMPARE",
          "widget": {
            "name": "compare_view"
          }
        }
      ],
      "mode": 0,
      "order": 5,
      "outputs": [],
      "pos": [
        300,
        2900
      ],
      "properties": {
        "Node name for S&R": "ImageCompare"
      },
      "size": [
        450,
        420
      ],
      "type": "ImageCompare",
      "widgets_values": [],
      "title": "Interactive A/B Comparison (Original Kid B vs Kid A Swap)"
    },
    {
      "flags": {},
      "id": 199,
      "inputs": [
        {
          "link": null,
          "name": "noise_seed",
          "type": "INT",
          "widget": {
            "name": "noise_seed"
          }
        },
        {
          "link": null,
          "name": "lora_name",
          "type": "COMBO",
          "widget": {
            "name": "lora_name"
          }
        },
        {
          "link": null,
          "name": "unet_name",
          "type": "COMBO",
          "widget": {
            "name": "unet_name"
          }
        },
        {
          "link": null,
          "name": "clip_name",
          "type": "COMBO",
          "widget": {
            "name": "clip_name"
          }
        },
        {
          "link": null,
          "name": "vae_name",
          "type": "COMBO",
          "widget": {
            "name": "vae_name"
          }
        },
        {
          "link": 354,
          "localized_name": "target_image",
          "name": "target_image",
          "type": "IMAGE"
        },
        {
          "link": 601,
          "localized_name": "target_mask",
          "name": "target_mask",
          "type": "MASK"
        },
        {
          "link": 355,
          "localized_name": "source_ref_1",
          "name": "source_ref_1",
          "type": "IMAGE"
        },
        {
          "link": 602,
          "localized_name": "source_ref_2",
          "name": "source_ref_2",
          "type": "IMAGE"
        },
        {
          "link": 603,
          "localized_name": "source_ref_3",
          "name": "source_ref_3",
          "type": "IMAGE"
        },
        {
          "link": 605,
          "localized_name": "source_ref_1_mask",
          "name": "source_ref_1_mask",
          "type": "MASK"
        },
        {
          "link": 606,
          "localized_name": "source_ref_2_mask",
          "name": "source_ref_2_mask",
          "type": "MASK"
        },
        {
          "link": 607,
          "localized_name": "source_ref_3_mask",
          "name": "source_ref_3_mask",
          "type": "MASK"
        }
      ],
      "mode": 0,
      "order": 2,
      "outputs": [
        {
          "links": [
            377
          ],
          "localized_name": "composited",
          "name": "composited",
          "type": "IMAGE"
        },
        {
          "links": [
            378
          ],
          "localized_name": "stitched",
          "name": "stitched",
          "type": "IMAGE"
        },
        {
          "links": [
            358
          ],
          "localized_name": "target_orig",
          "name": "target_orig",
          "type": "IMAGE"
        },
        {
          "links": [
            604
          ],
          "localized_name": "raw_face",
          "name": "raw_face",
          "type": "IMAGE"
        }
      ],
      "pos": [
        -700,
        3030
      ],
      "properties": {
        "previewExposures": []
      },
      "size": [
        370,
        330
      ],
      "type": "217f3726-59bc-4170-9ab1-c622bb578338",
      "widgets_values": [
        788122026421577,
        "Alissonerdx__BFS-Best-Face-Swap__bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
        "flux-2-klein-9b.safetensors",
        "qwen_3_8b_fp8mixed.safetensors",
        "flux2-vae.safetensors"
      ]
    },
    {
      "flags": {},
      "id": 204,
      "inputs": [
        {
          "link": 377,
          "localized_name": "images",
          "name": "images",
          "type": "IMAGE"
        },
        {
          "link": null,
          "localized_name": "filename_prefix",
          "name": "filename_prefix",
          "type": "STRING",
          "widget": {
            "name": "filename_prefix"
          }
        },
        {
          "link": null,
          "localized_name": "format",
          "name": "format",
          "type": "COMFY_DYNAMICCOMBO_V3",
          "widget": {
            "name": "format"
          }
        },
        {
          "link": null,
          "localized_name": "bit_depth",
          "name": "format.bit_depth",
          "type": "COMBO",
          "widget": {
            "name": "format.bit_depth"
          }
        },
        {
          "link": null,
          "localized_name": "input_color_space",
          "name": "format.input_color_space",
          "type": "COMBO",
          "widget": {
            "name": "format.input_color_space"
          }
        }
      ],
      "mode": 0,
      "order": 3,
      "outputs": [
        {
          "links": [
            379
          ],
          "localized_name": "images",
          "name": "images",
          "type": "IMAGE"
        }
      ],
      "pos": [
        -200,
        2900
      ],
      "properties": {},
      "size": [
        450,
        354
      ],
      "type": "SaveImageAdvanced",
      "widgets_values": [
        "face_swap_kid_a_composite",
        "png",
        "8-bit",
        "sRGB"
      ],
      "title": "Save Final Face Swap (Pristine Composite)"
    },
    {
      "flags": {},
      "id": 205,
      "inputs": [
        {
          "link": 378,
          "localized_name": "images",
          "name": "images",
          "type": "IMAGE"
        },
        {
          "link": null,
          "localized_name": "filename_prefix",
          "name": "filename_prefix",
          "type": "STRING",
          "widget": {
            "name": "filename_prefix"
          }
        },
        {
          "link": null,
          "localized_name": "format",
          "name": "format",
          "type": "COMFY_DYNAMICCOMBO_V3",
          "widget": {
            "name": "format"
          }
        },
        {
          "link": null,
          "localized_name": "bit_depth",
          "name": "format.bit_depth",
          "type": "COMBO",
          "widget": {
            "name": "format.bit_depth"
          }
        },
        {
          "link": null,
          "localized_name": "input_color_space",
          "name": "format.input_color_space",
          "type": "COMBO",
          "widget": {
            "name": "format.input_color_space"
          }
        }
      ],
      "mode": 0,
      "order": 4,
      "outputs": [
        {
          "links": null,
          "localized_name": "images",
          "name": "images",
          "type": "IMAGE"
        }
      ],
      "pos": [
        -200,
        3350
      ],
      "properties": {},
      "size": [
        450,
        354
      ],
      "type": "SaveImageAdvanced",
      "widgets_values": [
        "face_swap_kid_a_stitched",
        "png",
        "8-bit",
        "sRGB"
      ],
      "title": "Save Multi-View Stitched Sheet"
    },
    {
      "flags": {},
      "id": 82,
      "inputs": [
        {
          "label": "Kid A Ref 2",
          "link": null,
          "localized_name": "image",
          "name": "image",
          "type": "COMBO",
          "widget": {
            "name": "image"
          }
        },
        {
          "link": null,
          "localized_name": "choose file to upload",
          "name": "upload",
          "type": "IMAGEUPLOAD",
          "widget": {
            "name": "upload"
          }
        }
      ],
      "mode": 0,
      "order": 2,
      "outputs": [
        {
          "links": [
            602
          ],
          "localized_name": "IMAGE",
          "name": "IMAGE",
          "type": "IMAGE"
        },
        {
          "links": [],
          "localized_name": "MASK",
          "name": "MASK",
          "type": "MASK"
        }
      ],
      "pos": [
        -1200,
        3800
      ],
      "properties": {
        "Node name for S&R": "LoadImage",
        "cnr_id": "comfy-core",
        "ver": "0.8.2"
      },
      "size": [
        320,
        480
      ],
      "title": "Kid A Reference 2 (3/4 Angle / Profile)",
      "type": "LoadImage",
      "widgets_values": [
        "child_2.png",
        "image"
      ]
    },
    {
      "flags": {},
      "id": 83,
      "inputs": [
        {
          "label": "Kid A Ref 3",
          "link": null,
          "localized_name": "image",
          "name": "image",
          "type": "COMBO",
          "widget": {
            "name": "image"
          }
        },
        {
          "link": null,
          "localized_name": "choose file to upload",
          "name": "upload",
          "type": "IMAGEUPLOAD",
          "widget": {
            "name": "upload"
          }
        }
      ],
      "mode": 0,
      "order": 3,
      "outputs": [
        {
          "links": [
            603
          ],
          "localized_name": "IMAGE",
          "name": "IMAGE",
          "type": "IMAGE"
        },
        {
          "links": [],
          "localized_name": "MASK",
          "name": "MASK",
          "type": "MASK"
        }
      ],
      "pos": [
        -1200,
        4350
      ],
      "properties": {
        "Node name for S&R": "LoadImage",
        "cnr_id": "comfy-core",
        "ver": "0.8.2"
      },
      "size": [
        320,
        480
      ],
      "title": "Kid A Reference 3 (Expression / Smiling)",
      "type": "LoadImage",
      "widgets_values": [
        "child_3.png",
        "image"
      ]
    },
    {
      "flags": {},
      "id": 206,
      "inputs": [
        {
          "link": 604,
          "localized_name": "images",
          "name": "images",
          "type": "IMAGE"
        },
        {
          "link": null,
          "localized_name": "filename_prefix",
          "name": "filename_prefix",
          "type": "STRING",
          "widget": {
            "name": "filename_prefix"
          }
        },
        {
          "link": null,
          "localized_name": "format",
          "name": "format",
          "type": "COMFY_DYNAMICCOMBO_V3",
          "widget": {
            "name": "format"
          }
        },
        {
          "link": null,
          "localized_name": "bit_depth",
          "name": "format.bit_depth",
          "type": "COMBO",
          "widget": {
            "name": "format.bit_depth"
          }
        },
        {
          "link": null,
          "localized_name": "input_color_space",
          "name": "format.input_color_space",
          "type": "COMBO",
          "widget": {
            "name": "format.input_color_space"
          }
        }
      ],
      "mode": 0,
      "order": 6,
      "outputs": [
        {
          "links": null,
          "localized_name": "images",
          "name": "images",
          "type": "IMAGE"
        }
      ],
      "pos": [
        -200,
        3800
      ],
      "properties": {},
      "size": [
        450,
        354
      ],
      "title": "Save Raw Denoised Face",
      "type": "SaveImageAdvanced",
      "widgets_values": [
        "face_swap_kid_a_raw",
        "png",
        "8-bit",
        "sRGB"
      ]
    }
  ],
  "revision": 0,
  "version": 0.4
}
EOF
echo "  -> Workflow deployed successfully!"

echo "===================================================================="
echo " [SUCCESS] Setup Complete! Launching ComfyUI on Port 8188..."
echo "===================================================================="

# Verify ComfyUI core modules and comfy_kitchen import cleanly before launching
python -c "import comfy_kitchen; import comfy.quant_ops; import comfy.ldm.modules.attention; print('  ✓ Verified ComfyUI core modules and comfy_kitchen load cleanly!')" 2>&1 || {
    echo "  [Fallback] comfy_kitchen import check failed. Uninstalling to guarantee 100% stable startup..."
    pip uninstall -y comfy_kitchen 2>/dev/null || true
}

# Launch ComfyUI listening on all interfaces for RunPod HTTP proxy
exec python main.py --listen 0.0.0.0 --port 8188 --preview-method auto
