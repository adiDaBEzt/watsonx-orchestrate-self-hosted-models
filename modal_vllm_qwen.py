import subprocess
import modal

app = modal.App("vllm-qwen-orchestrate")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
API_KEY = "qwen-orchestrate-token"

hf_cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.3",
        "huggingface_hub",
        "hf-transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # Disable flashinfer JIT sampler (bypasses nvcc requirement)
    })
)

@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=600,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.web_server(port=8000, startup_timeout=360)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", API_KEY,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--max-model-len", "16384",
        "--enforce-eager",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)

