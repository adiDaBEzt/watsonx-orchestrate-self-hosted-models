#!/usr/bin/env python3
"""Lightweight background keep-warm daemon for Modal vLLM Qwen 2.5 7B server."""

import os
import sys
import time
import httpx
from datetime import datetime

VLLM_URL = os.getenv("VLLM_HOST_URL", "https://aditya-krishnan28--vllm-qwen-orchestrate-serve.modal.run/v1")
API_KEY = os.getenv("VLLM_API_KEY", "qwen-orchestrate-token")
INTERVAL_SECONDS = int(os.getenv("KEEPWARM_INTERVAL", "120"))

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Keep-Warm Daemon active.")
print(f"Target Endpoint: {VLLM_URL}/chat/completions")
print(f"Ping Interval:   {INTERVAL_SECONDS}s")
sys.stdout.flush()

payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 1,
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

while True:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_t = time.time()
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{VLLM_URL}/chat/completions",
                json=payload,
                headers=headers
            )
            elapsed = time.time() - start_t
            if resp.status_code == 200:
                print(f"[{now_str}] PING OK - Status: 200 ({elapsed:.2f}s) - Warm")
            else:
                print(f"[{now_str}] PING WARN - Status: {resp.status_code} ({elapsed:.2f}s) - {resp.text[:100]}")
    except Exception as e:
        elapsed = time.time() - start_t
        print(f"[{now_str}] PING PENDING/TIMEOUT ({elapsed:.2f}s): {e}")

    sys.stdout.flush()
    time.sleep(INTERVAL_SECONDS)
