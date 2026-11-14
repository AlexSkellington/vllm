#!/bin/bash
# Backport of vllm-project/vllm#59973 (merged 2026-10-15, after the v0.32.0 branch cut): opt-in quantized private copy
# of the shared lm_head for MTP/EAGLE drafters ("draft_lm_head_quantization": "nvfp4" | "fp8" in --speculative-config).
# The target still verifies with its bf16 head (outputs unchanged; only acceptance length can move). On sm_86 use
# "nvfp4" (weight-only Marlin W4A16); "fp8" needs FP8 activations (sm_89+). Usage: apply_vllm_pr59973_backport.sh VENV
set -e; SP=$(ls -d $1/lib/python3*/site-packages)
grep -q draft_lm_head_quantization $SP/vllm/config/speculative.py && { echo "already applied"; exit 0; }
cd $SP && patch -p1 --dry-run < /root/backports/pr59973-vllm.diff >/dev/null && patch -p1 < /root/backports/pr59973-vllm.diff
