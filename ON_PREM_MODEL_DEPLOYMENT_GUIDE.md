# Enterprise Architecture Guide: Deploying Self-Hosted & On-Prem Models with IBM watsonx Orchestrate (WXO)

This guide provides a comprehensive architecture blueprint, prerequisite checklist, endpoint conventions, and step-by-step configuration workflows for enterprise clients hosting Large Language Models (LLMs) on-premises (or in private clouds) and integrating them seamlessly into **IBM watsonx Orchestrate**.

---

## 1. Executive Summary & Architecture Overview

In an enterprise environment, hosting open-weight models (such as **Qwen 2.5**, **Llama 3.3**, or **Granite**) on private infrastructure is standard practice for data sovereignty, regulatory compliance, and cost control. 

watsonx Orchestrate supports custom LLMs via its **Virtual Model** abstraction layer. There are two primary serving paradigms:

1. **vLLM (OpenAI-Compatible)**: High-throughput, production-grade engine with PagedAttention, continuous batching, and native Hermes tool-calling parser support.
2. **Ollama**: Lightweight, GGUF-quantized inference server ideal for rapid deployments, edge nodes, and developer sandboxes.

### Network Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IBM watsonx Orchestrate Control Plane                  │
│                     (SaaS / Cloud / On-Prem OpenShift CP4D)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                HTTPS TLS 1.3 / mTLS   │ (Encrypted Payload over Port 443)
                API Key / Bearer Auth  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Enterprise Ingress / API Gateway Layer                   │
│         (F5 BIG-IP / Kong / Envoy / Apigee / Cloudflare / Reverse Proxy)    │
│  - Public DNS (if SaaS WXO) or Internal Corporate DNS (if On-Prem WXO)      │
│  - Edge TLS Termination & WAF Inspection                                    │
│  - Header Mapping & Rate Limiting                                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Private Corporate LAN / VPC)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Enterprise On-Prem Compute (OpenShift / Kubernetes)            │
│                                                                             │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │     vLLM Pods / Service         │   │     Ollama Pods / Service       │  │
│  │  - Engine: vLLM 0.28+           │   │  - Engine: Ollama 0.33+         │  │
│  │  - Model: Qwen/Qwen2.5-7B-Inst  │   │  - Model: qwen2.5:7b (GGUF)     │  │
│  │  - Tool Parser: hermes          │   │  - Endpoint: /v1, /api          │  │
│  │  - Port: 8000                   │   │  - Port: 11434                  │  │
│  │  - GPUs: NVIDIA A100/H100/L40S  │   │  - GPUs: NVIDIA L4/A10/A30      │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites Checklist

Before configuring watsonx Orchestrate, the client's infrastructure/MLOps team must prepare the following:

### Infrastructure & Hardware
- [ ] **GPU Infrastructure**: Minimum 1x 24GB VRAM GPU (e.g. NVIDIA L4, A10G, A5000) for a 7B FP16/BF16 model; 2x-4x 80GB A100/H100 for 70B+ models using Tensor Parallelism (`-tp`).
- [ ] **Container Platform**: Red Hat OpenShift, Kubernetes, Docker Engine, or Bare-Metal Linux (Ubuntu 22.04 LTS / RHEL 9).
- [ ] **Model Weights**: Safetensors downloaded from Hugging Face or internal model registry (e.g. Artifactory / MinIO).

### Networking & Security
- [ ] **Ingress / Reverse Proxy**: F5, Kong, NGINX, or Cloudflare configured with a valid public TLS certificate (Let's Encrypt, DigiCert, corporate CA).
- [ ] **Firewall / Inbound Rules**:
  - If using **WXO SaaS**: Endpoint must resolve publicly over HTTPS (Port 443).
  - If using **WXO On-Premises (CP4D)**: Endpoint must be resolvable from the WXO cluster pods.
- [ ] **Authentication**: An API Bearer Token / Header Secret configured on the gateway or passed directly to vLLM via `--api-key`.

---

## 3. Serving Options: vLLM vs Ollama

| Criterion | Option A: vLLM (Recommended for Enterprise) | Option B: Ollama (Recommended for Edge / Dev) |
| :--- | :--- | :--- |
| **Primary Use Case** | High-concurrency production agent serving | Low-latency single-user tasks, testing, edge |
| **Model Format** | Full precision (BF16, FP8, AWQ, GPTQ) safetensors | GGUF Quantized (Q4_K_M, Q8_0) |
| **WXO Provider Type** | `openai` (`virtual-model/openai/<model-name>`) | `ollama` (`virtual-model/ollama/<model-name>`) |
| **WXO Authentication** | External Connection (`connections` CLI command) | Static `api_key` in model YAML |
| **Tool Calling / Function Calling** | Native `--tool-call-parser hermes` (High Accuracy) | Built-in template tool parser |
| **Batching & Concurrency** | PagedAttention + Continuous Batching (High Throughput) | Sequential / limited slot concurrency |

---

## 4. API Endpoints & Contract Conventions

When watsonx Orchestrate sends requests to a custom model, it adheres to the following conventions:

### vLLM (OpenAI-Compatible Provider)
- **Base Host URL format**: `https://<domain>` (do **not** append `/v1` or `/v1/chat/completions` in the YAML).
- **Inference Route**: WXO automatically appends `/v1/chat/completions`.
- **Payload Schema**: Standard OpenAI Chat Completions JSON (`messages`, `tools`, `temperature`, `max_tokens`, `stream`).
- **Tool Calling Support**: Requires vLLM to be started with:
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --api-key <YOUR_SECRET_API_KEY> \
    --host 0.0.0.0 --port 8000
  ```

### Ollama Provider
- **Base Host URL format**: `https://<domain>` (do **not** append `/api` or `/v1`).
- **Inference Route**: WXO contacts the native Ollama OpenAI-compatible routes directly (`/v1/chat/completions` or `/api/chat`).
- **Server Startup Command**:
  ```bash
  OLLAMA_HOST=0.0.0.0:11434 ollama serve
  ollama pull qwen2.5:7b
  ```

---

## 5. Step-by-Step Client Configuration in watsonx Orchestrate

### Step 5.1: Configure Authentication

For vLLM and OpenAI-compatible providers, authentication credentials must be configured for **both** the `draft` and `live` environments.

```bash
# 1. Create connection in Draft environment
orchestrate connections set-credentials \
  --name qwen_modal_creds \
  --env draft \
  --data '{"api_key": "YOUR_ENTERPRISE_API_KEY"}'

# 2. Create connection in Live environment
orchestrate connections set-credentials \
  --name qwen_modal_creds \
  --env live \
  --data '{"api_key": "YOUR_ENTERPRISE_API_KEY"}'
```

*(Note: For Ollama, WXO uses the `api_key` directly from the model YAML spec, so no external connection object is needed).*

---

### Step 5.2: Create & Import the Model Spec YAML

#### For vLLM (`models/qwen2.5-vllm.yaml`):
```yaml
spec_version: v1
kind: model
name: virtual-model/openai/Qwen/Qwen2.5-7B-Instruct
display_name: Qwen 2.5 7B Instruct (Enterprise vLLM)
description: |
  Enterprise self-hosted Qwen 2.5 7B model running on on-prem vLLM infrastructure.
tags:
  - qwen
  - vllm
  - on-prem
model_type: chat
provider_config:
  custom_host: https://llm-gateway.corp.domain.com
  connection: qwen_modal_creds
config:
  max_completion_tokens: 8192
  temperature: 0.0
```

#### For Ollama (`models/qwen2.5-ollama.yaml`):
```yaml
spec_version: v1
kind: model
name: virtual-model/ollama/qwen2.5:7b
display_name: Qwen 2.5 7B (Enterprise Ollama)
description: |
  Enterprise self-hosted Qwen 2.5 7B model running on on-prem Ollama infrastructure.
tags:
  - qwen
  - ollama
  - on-prem
model_type: chat
provider_config:
  custom_host: https://ollama-gateway.corp.domain.com
  api_key: ollama
config:
  max_completion_tokens: 4096
  temperature: 0.0
```

#### Import the Model:
```bash
orchestrate models import --file models/qwen2.5-vllm.yaml
```

---

### Step 5.3: Run the Official Capability Test Suite

watsonx Orchestrate provides an automated validation engine to guarantee that your on-prem model supports required agentic capabilities:

```bash
orchestrate models validate --name virtual-model/openai/Qwen/Qwen2.5-7B-Instruct --verbose
```

The CLI executes four automated tests:
1. `basic_invocation`: Single-turn prompt/completion.
2. `streaming`: Server-Sent Events (SSE) streaming token verification.
3. `tool_calling`: Validates whether the model outputs JSON arguments conforming to function definitions.
4. `streaming_tool_calling`: Validates streaming function calls during execution.

*Expected Result: 4/4 Passed (100.0% Success Rate).*

---

### Step 5.4: Bind the Model to an Orchestrate Agent

Create the agent definition (`agents/enterprise-assistant.yaml`):

```yaml
spec_version: v1
kind: native
name: enterprise_qwen_assistant
llm: virtual-model/openai/Qwen/Qwen2.5-7B-Instruct
style: react_core
hide_reasoning: false
memory_enabled: true
description: |
  Enterprise assistant powered by on-premises Qwen 2.5.
instructions: |
  You are an enterprise AI assistant running securely on corporate infrastructure.
  Respond accurately and concisely. Use tools whenever available.
tools: []
collaborators: []
```

#### Deploy and Chat:
```bash
# Import agent to Draft
orchestrate agents import --file agents/enterprise-assistant.yaml

# Deploy to Live
orchestrate agents deploy --name enterprise_qwen_assistant

# Test chat from CLI
orchestrate chat ask --agent-name enterprise_qwen_assistant "Hello! Confirm your model and host."
```

---

## 6. Enterprise Production Best Practices

1. **High Availability & Load Balancing**:
   - Run at least 2 replica pods behind an Ingress Controller (e.g. NGINX or HAProxy) with round-robin or least-connections routing.
   - Use health check endpoints (`/health` on vLLM, `/api/tags` on Ollama).
2. **Timeouts**:
   - Long reasoning chains or large context windows require sufficient HTTP timeouts. Configure proxy timeouts (e.g. `proxy_read_timeout 300s`) so reverse proxies do not drop streaming connections prematurely.
3. **mTLS & VPC Peering**:
   - If using watsonx Orchestrate on IBM Cloud and models in AWS/Azure/Equinix on-prem, use IBM Cloud Direct Link or VPN / Transit Gateway peering to ensure traffic never traverses public internet unencrypted.
4. **Context Length Sizing**:
   - Set `max_model_len` in vLLM (e.g. 8192 or 16384) according to available GPU VRAM. Ensure KV cache memory does not exceed available physical VRAM to prevent OOM errors under peak concurrency.
