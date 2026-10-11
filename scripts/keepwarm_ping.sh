#!/usr/bin/env bash
# ==============================================================================
# keepwarm_ping.sh
# Sends lightweight ping requests to Modal / on-prem inference endpoints
# to keep GPU containers warm and avoid cold starts during live sessions.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env if present
if [ -f "${ROOT_DIR}/.env" ]; then
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/.env"
fi

VLLM_URL="${VLLM_HOST_URL:-https://aditya-krishnan28--vllm-qwen-orchestrate-serve.modal.run/v1}"
OLLAMA_URL="${OLLAMA_HOST_URL:-https://aditya-krishnan28--ollama-qwen-orchestrate-serve.modal.run}"
VLLM_TOKEN="${VLLM_API_KEY:-qwen-orchestrate-token}"

INTERVAL_SECONDS="${1:-180}"  # Default to 180s (3 minutes)

echo "Starting Keep-Warm Ping Loop (Interval: ${INTERVAL_SECONDS}s)"
echo "vLLM Target:   ${VLLM_URL}/chat/completions"
echo "Ollama Target: ${OLLAMA_URL}/v1/chat/completions"
echo "Press [Ctrl+C] to exit."
echo "------------------------------------------------------------------"

ping_once() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] Pinging endpoints..."

    # Ping vLLM
    vllm_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${VLLM_URL}/chat/completions" \
        -H "Authorization: Bearer ${VLLM_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}' || echo "TIMEOUT")
    echo "  -> vLLM response status: ${vllm_code}"

    # Ping Ollama
    ollama_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "${OLLAMA_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}' || echo "TIMEOUT")
    echo "  -> Ollama response status: ${ollama_code}"
}

while true; do
    ping_once
    sleep "${INTERVAL_SECONDS}"
done
