#!/usr/bin/env bash
# ==============================================================================
# setup_orchestrate.sh
# Automated Setup Script for IBM watsonx Orchestrate (WXO)
# Registers Custom Qwen Models, Tavily Remote MCP Toolkit, and Deploys Agents
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env if present
if [ -f "${ROOT_DIR}/.env" ]; then
    echo "[INFO] Loading configuration from .env..."
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/.env"
fi

# Detect orchestrate executable
if [ -f "${ROOT_DIR}/venv/bin/orchestrate" ]; then
    ORCHESTRATE="${ROOT_DIR}/venv/bin/orchestrate"
elif command -v orchestrate >/dev/null 2>&1; then
    ORCHESTRATE="orchestrate"
else
    echo "[ERROR] orchestrate CLI not found. Please activate your venv or install ibm-watsonx-orchestrate."
    exit 1
fi

echo "======================================================================"
echo " IBM watsonx Orchestrate - Self-Hosted Models & MCP Setup"
echo "======================================================================"

# 1. Environment Activation Check
echo ""
echo "[Step 1/5] Checking Orchestrate active environment..."
if [ -n "${WXO_INSTANCE_URL:-}" ] && [ -n "${WXO_API_KEY:-}" ]; then
    ENV_NAME="${WXO_ENV_NAME:-qwen-env}"
    echo "[INFO] Adding/activating environment '${ENV_NAME}'..."
    ${ORCHESTRATE} env add -n "${ENV_NAME}" -u "${WXO_INSTANCE_URL}" -t ibm_iam 2>/dev/null || true
    ${ORCHESTRATE} env activate "${ENV_NAME}" --api-key "${WXO_API_KEY}"
else
    echo "[INFO] Using currently active Orchestrate environment."
    ${ORCHESTRATE} env list || true
fi

# 2. Connection and Credentials Setup
APP_ID="${CONNECTION_APP_ID:-qwen_modal_creds}"
VLLM_KEY="${VLLM_API_KEY:-qwen-orchestrate-token}"

echo ""
echo "[Step 2/5] Configuring credentials connection '${APP_ID}'..."
${ORCHESTRATE} connections add --app-id "${APP_ID}" 2>/dev/null || true
${ORCHESTRATE} connections configure --app-id "${APP_ID}" --env draft --type team --kind key_value 2>/dev/null || true
${ORCHESTRATE} connections configure --app-id "${APP_ID}" --env live --type team --kind key_value 2>/dev/null || true
${ORCHESTRATE} connections set-credentials --app-id "${APP_ID}" --env draft -e "api_key=${VLLM_KEY}"
${ORCHESTRATE} connections set-credentials --app-id "${APP_ID}" --env live -e "api_key=${VLLM_KEY}"
echo "[✓] Connection '${APP_ID}' configured for draft and live environments."

# 3. Model Import & Validation
echo ""
echo "[Step 3/5] Importing and validating Virtual Models..."
if [ -f "${ROOT_DIR}/models/qwen2.5-modal.yaml" ]; then
    echo "[INFO] Importing vLLM model: Qwen/Qwen2.5-7B-Instruct..."
    ${ORCHESTRATE} models import --file "${ROOT_DIR}/models/qwen2.5-modal.yaml" --app-id "${APP_ID}"
fi

if [ -f "${ROOT_DIR}/models/qwen2.5-ollama.yaml" ]; then
    echo "[INFO] Importing Ollama model: qwen2.5:7b..."
    ${ORCHESTRATE} models import --file "${ROOT_DIR}/models/qwen2.5-ollama.yaml"
fi
echo "[✓] Virtual models validated successfully."

# 4. Tavily MCP Toolkit Import
echo ""
echo "[Step 4/5] Importing Tavily MCP Toolkit..."
TAVILY_FILE="${ROOT_DIR}/toolkits/tavily-search.yaml"
if [ -n "${TAVILY_API_KEY:-}" ] && [ -f "${ROOT_DIR}/toolkits/tavily-search.yaml.template" ]; then
    echo "[INFO] Generating toolkit YAML from template using TAVILY_API_KEY..."
    sed "s|\${TAVILY_API_KEY}|${TAVILY_API_KEY}|g" "${ROOT_DIR}/toolkits/tavily-search.yaml.template" > "${TAVILY_FILE}"
fi

if [ -f "${TAVILY_FILE}" ]; then
    ${ORCHESTRATE} toolkits import --file "${TAVILY_FILE}"
    echo "[✓] Tavily MCP toolkit imported."
else
    echo "[WARN] Toolkit file not found. Skipping."
fi

# 5. Agent Import and Deployment
echo ""
echo "[Step 5/5] Importing and Deploying Agents..."
if [ -f "${ROOT_DIR}/agents/qwen-assistant.yaml" ]; then
    ${ORCHESTRATE} agents import --file "${ROOT_DIR}/agents/qwen-assistant.yaml"
    ${ORCHESTRATE} agents deploy --name qwen_simple_agent
    echo "[✓] qwen_simple_agent deployed to live."
fi

if [ -f "${ROOT_DIR}/agents/qwen-ollama-agent.yaml" ]; then
    ${ORCHESTRATE} agents import --file "${ROOT_DIR}/agents/qwen-ollama-agent.yaml"
    ${ORCHESTRATE} agents deploy --name qwen_ollama_agent
    echo "[✓] qwen_ollama_agent deployed to live."
fi

echo ""
echo "======================================================================"
echo " [SUCCESS] All components configured and deployed!"
echo " Test with:"
echo "   orchestrate chat ask -n qwen_simple_agent \"Search the web for the latest NASA news\""
echo "   orchestrate chat ask -n qwen_ollama_agent \"Search the web for Tokyo weather\""
echo "======================================================================"
