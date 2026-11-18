#!/bin/bash
# vLLM engine for AS1 CT100 (2026-10-28). Served on 127.0.0.1:8000 ONLY; the lab talks to the llama.cpp-compatible
# gateway on :8081 (vllm-direct.service, /opt/vllm/vllm_gateway.py). See /opt/vllm/README.md.
#  - FlashInfer + the checkpoint's calibrated FP8 KV (default): the only combo whose long-context decode holds up
#    (bf16 KV on either backend: 32k decode 53-57 tok/s vs 102; FLASH_ATTN cannot do FP8 KV on sm_86).
#  - 2026-11-08: draft_sample_method probabilistic (was greedy default): sampled single 128.3 -> 137.6 tok/s (+7.3%), pairs 223.7 -> 241.0 (+7.8%), acceptance 0.584 -> 0.671 (venv_ab.py probonly ABBA). Rejection sampling is exact. Rollback: /root/backports/vllm-engine.sh.bak-probdraft-*
#  - MTP 2 (3 = no single-stream gain, 943 -> 683 tok/s at 32 streams). Drafter uses a private NVFP4 copy of the shared
#    bf16 lm_head (vllm#59973 backport, 2026-11-01): decode +7.8%, 32k +7.2%, edits wall -11%, KV pool -3.7%.
#  - --prefix-match-unit 32 + --enable-mamba-shared-prefix-checkpoint: per-turn agent TTFT 1.98 -> 0.86 s @30k,
#    2.93 -> 1.26 s @100k (hybrid GDN state is only checkpointed on 1,600-token blocks otherwise).
#  - --enable-prompt-tokens-details: cached_tokens in usage -> llama.cpp-style timings.cache_n in the gateway.
#  - 2026-11-02: vLLM main nightly 0.33.0rc1.dev452+g8530a778f (/root/vllm-venv-main: +#64999, PCIe-IPC cap, split-K re-targeted
#    to cuda_unquantized_gemm, #59973 native). VLLM_USE_BREAKABLE_CUDAGRAPH=0: Qwen3.5 auto-enables breakable graphs on
#    NVIDIA (KV 509k -> 408k and slower). vs 0.32: decode +11%, 32k +14%, edits -13.6%, prefill flat (5k -2.9%), TTFT +40 ms,
#    KV 485,562; full GSM8K c1 1264 vs 1268/1319 (4 vs 8 flips, p~0.4). Rollback: /root/backports/*.bak-pre-main-*.
#  - 2026-11-02: two full 262k slots back: CUDA graphs captured only up to 128 tokens with main's default small sizes kept
#    (3/6/12 matter: 4 streams x 3 MTP = 12 = VLLM_PCIE_IPC_MAX_TOKENS; a bare max_cudagraph_capture_size=128 drops them
#    and costs 4 streams -11%). Graph pool 2.07 -> 0.76 GiB, peak activation 3.17 -> 2.11 GiB, KV 485,562 -> 543,650 (2.07x).
#    32k decode ABA 130.2 vs 129.5/126.3, 100k 122.8 vs 119.6/118.3, 1-32 streams equal. Rollback: *.bak-pre-graphlist-*.
#  - Output ceiling comes from the shared router policy; do not add a per-engine numeric ceiling.
#  - 2026-11-10: tiered KV offload (OffloadingConnector/TieringOffloadingSpec: 16 GiB CPU tier in /dev/shm + NVMe fs
#    tier /var/cache/vllm-kvcache, dataset Samsung_PM9A1_1TB/vllm-kvcache) = chats survive engine restarts: 30k chat
#    12.87 s -> 2.23 s, identical output. --max-num-seqs 64 (default 128) frees the probabilistic-draft buffers:
#    KV 519,819 -> 552,587 (2.1x 262k). A/B kvoff2: decode +0.55%, pairs +0.35%, prefill flat. Trim: vllm-kvcache-trim.timer.
#    Rollback: /opt/vllm/vllm-engine.sh.bak-pre-kvoff-20261109-205558
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 HOME=/root PATH=/root/vllm-venv-main/bin:/usr/bin:/bin
# Fetch the authoritative policy; generated last-good cache allows an independent boot.
GENERATION_OVERRIDE=$(/root/vllm-venv-main/bin/python /opt/vllm/shared_generation_policy.py) || exit 1
exec /root/vllm-venv-main/bin/vllm serve /root/models-vllm/qwen38-int4 \
  --served-model-name Qwen3.8-27B-INT4 --tensor-parallel-size 2 \
  --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_lm_head_quantization":"nvfp4","draft_sample_method":"probabilistic"}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --prefix-match-unit 32 --enable-mamba-shared-prefix-checkpoint --enable-prompt-tokens-details --enable-per-request-metrics \
  --override-generation-config "$GENERATION_OVERRIDE" \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]}' --host 127.0.0.1 --port 8000 --disable-uvicorn-access-log --max-num-seqs 64 --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":17179869184,"secondary_tiers":[{"type":"fs","root_dir":"/var/cache/vllm-kvcache"}]}}'
