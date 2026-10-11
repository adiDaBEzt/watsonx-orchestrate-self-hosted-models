import subprocess
import time
import os
import modal

app = modal.App("ollama-qwen-orchestrate")

# Modal container image with Ollama and required tools
ollama_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({"OLLAMA_HOST": "0.0.0.0:11434"})
)

# Persistent volume so the pulled Qwen weights persist across cold starts
ollama_volume = modal.Volume.from_name("ollama-models-cache", create_if_missing=True)

MODEL_NAME = "qwen2.5:7b"

@app.function(
    image=ollama_image,
    gpu="L4",               # NVIDIA L4 GPU
    timeout=600,
    scaledown_window=300,   # Keep warm for 5 minutes of inactivity, then scale to 0
    volumes={"/root/.ollama": ollama_volume},
)
@modal.web_server(port=11434, startup_timeout=360)
def serve():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"
    
    # 1. Start Ollama server
    proc = subprocess.Popen(["ollama", "serve"], env=env)
    
    # 2. Wait for Ollama server to become responsive
    for _ in range(30):
        try:
            res = subprocess.run(["ollama", "list"], env=env, capture_output=True, text=True)
            if res.returncode == 0:
                break
        except Exception:
            pass
        time.sleep(1)
        
    # 3. Pull Qwen 2.5 7B (if already in volume, finishes in seconds)
    subprocess.run(["ollama", "pull", MODEL_NAME], env=env, check=True)

