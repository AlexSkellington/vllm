# vLLM on Ai-Server-1 (CT100, 2x RTX 3090) — since 2026-10-28

AS1 serves **Qwen3.8-27B-INT4** with **vLLM 0.32.0** instead of llama.cpp. The rest of the lab (CT108 router,
CT107 watchdog/monitor/lab map, CT101 MCP model tools, OpenCode, Open WebUI) talks to it exactly as it talked to
llama-server, through a llama.cpp-compatible gateway. llama.cpp stays installed (disabled) as the rollback.

```
lab (:8081) --> vllm-direct.service  /opt/vllm/vllm_gateway.py   (llama.cpp-compatible front, BindsTo the engine)
                     |
                     v
           vllm-engine.service  /opt/vllm/vllm-engine.sh  ->  vllm serve on 127.0.0.1:8000 ONLY
           ExecStartPost: /usr/local/bin/vllm-envcheck   (expectations in /opt/vllm/engine.expect)
```

## Files

| Path | What |
|---|---|
| `/opt/vllm/vllm-engine.sh` | the engine command line, with the reason for every flag |
| `/opt/vllm/vllm_gateway.py` | the gateway (docstring = full contract) |
| `/opt/vllm/engine.expect` | venv / vLLM / torch / CUDA / GPU count that vllm-envcheck asserts |
| `/usr/local/bin/vllm-envcheck` | vLLM analogue of llamacpp-libcheck: `vllm-envcheck: OK ...` in the engine journal at every start |
| `/etc/systemd/system/vllm-engine.service` (+ `.d/98-envcheck.conf`), `vllm-direct.service` | units, both enabled |
| `/root/vllm-venv-032` | Python 3.12 venv: vllm 0.32.0, torch 2.13.0+cu130 (old 0.29 test venv: `/root/vllm-venv`) |
| `/root/models-vllm/qwen38-int4` | RedHatAI Qwen3.8-27B INT4 (W4A16 g128, compressed-tensors) + calibrated FP8 KV, MTP head bf16 |
| `/root/engbench.py`, `/root/agentsim.py`, `/root/reread.py`, `/root/gsm.py`, `/root/gwtest.py` | benchmark / test harnesses |

## What the gateway provides (so nothing downstream needs a vLLM code path)

`/health` (503 while loading), `/props` (n_ctx, total_slots=16, chat_template, modalities, `build_info`,
`engine`, `engine_unit`, `engine_args` = the engine's LIVE argv), `/slots` (one entry per in-flight request:
`is_processing`, `n_prompt_tokens(_processed/_cache)`, `next_token[0].n_decoded`, `n_ctx`, `id_task`),
`/v1/models` with the llama.cpp `meta` block, `/tokenize`, `/detokenize`, `/completion`, `/models/load`
(no-op success for the resident model), `/models/unload` (refused: single-model engine), `/metrics` (vLLM's).
Generation: strips llama.cpp-only fields (`samplers`, `cache_prompt`, `id_slot`, ...), maps `n_predict` and
`repeat_penalty`, renames vLLM's `delta.reasoning` to `reasoning_content`, adds llama.cpp `timings` (incl.
`cache_n`, `draft_n`, `draft_n_accepted`) and `prompt_progress` frames. Non-stream requests are streamed
upstream and re-assembled (tool calls included) so they get the same timings and live /slots progress.

**Prefill progress is an estimate.** vLLM reports nothing until the first token (its counters only move when a
request finishes - measured). `n_prompt_tokens_processed` = elapsed x learned prefill rate, capped one short of
the total; once capped it advances +1/s only while the GPUs are busy and the request is younger than 4x its
expected prefill time. So a hung engine freezes the marker and the CT107 watchdog's busy-but-frozen logic works.
Decode counts are exact (continuous usage stats).

## Measured (2026-10-28, GPU unlock ON)

| | value |
|---|---|
| single-stream decode (400 tok, no-think) | ~118-120 tok/s (MTP accept len ~2.1-2.4) |
| prefill 4k / 32k / 64k / 100k / 160k / 241k | 2,608 / 2,372 / 2,146 / 1,936 / 1,665 / 1,388 tok/s |
| decode at 32k / 100k / 241k context | 102.6 / 85.4 / 83.2 tok/s |
| agent turn (+1.5k tokens) at 30k / 100k | TTFT 0.86 s / 1.26 s (was 1.98 / 2.93 s before prefix-match-unit) |
| concurrency 2 / 4 / 8 / 16 / 32 streams | 188 / 366 / 577 / 835 / 951 tok/s aggregate (through the gateway) |
| gateway overhead | -0.5% single stream, -0.7% at 32 streams |
| GSM8K (250, greedy, no-think) | 97.6% |
| cold start to healthy | 80-290 s (watchdog grace for vLLM: 600 s) |

## Operating

```bash
systemctl status vllm-engine vllm-direct           # both should be active
journalctl -u vllm-engine -b | grep vllm-envcheck  # OK line at every start
curl -s 127.0.0.1:8081/props | python3 -m json.tool | head -40
curl -s 127.0.0.1:8081/slots | python3 -m json.tool | head -30
systemctl restart vllm-engine                      # restarts the gateway too (BindsTo)
```
The CT107 watchdog recovers vLLM by **restarting vllm-engine.service** (KillMode=control-group takes the TP workers
and their GPU memory with it) after saving `/root/hang-capture-<time>.txt` (GPU state, /slots, engine + gateway
journals). Never `pkill -f vllm...` from a shell whose own command line contains the pattern: it kills itself.

## Upgrading vLLM

1. New venv next to the old one (`uv venv -p 3.12 /root/vllm-venv-NNN && uv pip install --python ... vllm==X`).
2. Point `vllm-engine.sh` (PATH + exec) and `vllm-direct.service` (ExecStart python) at it; update `engine.expect`.
3. `systemctl daemon-reload && systemctl restart vllm-engine`; check `vllm-envcheck: OK`, `/props`, then rerun
   `/root/engbench.py`, `/root/agentsim.py` and `/root/gsm.py` against :8081 and compare with the table above.
4. Re-apply the lab patches: `/opt/vllm/apply_vllm_pcieipc_cap.py /root/vllm-venv-NNN` (see All-reduce below) and
   `/opt/vllm/apply_vllm_skinny_splitk.py /root/vllm-venv-NNN` (see Skinny GEMV below).
   On a vLLM that predates vllm#59973 also `/opt/vllm/apply_vllm_pr59973_backport.sh /root/vllm-venv-NNN` (Draft head
   below); from the first release that contains #59973 the key works natively and the backport is not needed.
5. Check the vLLM release notes for: the hybrid prefix-cache flags (`--prefix-match-unit`,
   `--enable-mamba-shared-prefix-checkpoint`), the `reasoning` vs `reasoning_content` field, `continuous_usage_stats`.

## Rollback to llama.cpp (≈2 min)

```bash
systemctl disable --now vllm-direct vllm-engine
systemctl enable --now llamacpp-direct      # runtime ...-qr2-vp-gdn3 (0023), unchanged
```
then on CT101 rename `vllm-3090x2` back to `llamacpp-3090x2` in `/opt/mcp/instance_policy.json`, on CT108 set
`ROUTER_EXCLUDE_INSTANCES=llamacpp-3090x2` in `model-router.service.d/80-exclude-3090.conf` (restart model-router),
and move `Qwen3.8-27B-UD-Q4_K_XL` back out of `profile_policy.retired` in `/opt/model-control/sampling_profiles.json`.
Discovery, watchdog, monitor and the MCP sync need nothing: they handle both kinds.

## Engine flags that matter (and the measured reasons)

- **FlashInfer + the checkpoint's FP8 KV** (defaults). bf16 KV on either backend: 132 tok/s short decode but
  long-context decode collapses (32k: 53-57, 100k: 24-27 tok/s) and capacity halves; FLASH_ATTN cannot use FP8 KV
  on sm_86.
- **MTP 2 draft tokens.** 3 = no single-stream gain, 943 -> 683 tok/s at 32 streams.
- **`--prefix-match-unit 32 --enable-mamba-shared-prefix-checkpoint`.** Qwen3.8's GDN state is only checkpointed on
  1,600-token blocks (vLLM sizes the attention block to the mamba page), so every turn re-computed up to a block of
  old history: agent-turn TTFT 1.98 -> 0.86 s at 30k, 2.93 -> 1.26 s at 100k.
- **`--enable-prompt-tokens-details`** gives the gateway `cached_tokens` (llama.cpp `timings.cache_n`).
- **`--override-generation-config '{"max_new_tokens":16384}'`** mirrors llama.cpp's `--n-predict 16384`; the
  checkpoint's own defaults (temp 1.0, top-k 20, top-p 0.95) already equal llama.cpp production's.
- `--max-num-batched-tokens 8192`: +1% prefill only and needs a shorter max length - not used.

## Capabilities are derived, never hand-listed (2026-10-31)

The gateway advertises only what the checkpoint and the running engine actually provide, the way
llama-server derives /props from its GGUF, mmproj and template probe. A different checkpoint or engine
flag set changes the advertised capabilities with no gateway edit.

| Advertised | Source |
|---|---|
| checkpoint dir, `model_path` | engine live argv (`vllm serve <dir>`); `GW_MODEL_DIR` is only an optional override (removed from the unit) |
| `chat_template_caps` | template rendered with probe messages through transformers' sandboxed Jinja (vLLM's renderer); same probes as llama.cpp `common/jinja/caps.cpp` - identical results on all three llama.cpp templates in the lab |
| tools / tool calls / parallel | template caps AND `--enable-auto-tool-choice` + `--tool-call-parser` in the live argv |
| `modalities` image / video / audio | checkpoint encoder config + processor, minus `--language-model-only` or a zero `--limit-mm-per-prompt` entry (image + video verified with real requests) |
| accepted request fields | engine `/openapi.json` (`ChatCompletionRequest` / `CompletionRequest`); engine_compat's pinned 0.32 lists are only the fallback |
| `model_ftype`, speculative type | checkpoint `quantization_config` + `--kv-cache-dtype`; `--speculative-config` |
| `preserve_thinking` etc. | applied by default when the template reads them, like llama-server's default preserve_reasoning; client kwargs win |

The CT108 router passes this contract through, publishes per-model `tools` in `/_router/tps` `effort`,
and waits (ROUTER_EFFORT_LOAD_WAIT_S, 300 s) for a loading model's template instead of returning 503.
OpenCode's `reasoning` / `tool_call` come from that probe (opencode-genconfig.py `model_capabilities`).
Backups of the previous files: /root/auto-caps-20261031/.

## All-reduce: FlashInfer PCIe IPC for small batches (2026-10-31)

`vllm-engine.service.d/60-flashinfer-pcie-ipc-ar.conf` sets `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` and
`VLLM_PCIE_IPC_MAX_TOKENS=12`. The second variable comes from a **lab-local vLLM patch**,
`/opt/vllm/apply_vllm_pcieipc_cap.py VENV` (idempotent; backup `*.bak-pre-lab-cap`). Upstream routes every captured
batch (up to 512 tokens) through PCIe IPC, which wins small and loses big; the cap keeps only the winning range
(12 tokens = 4 streams x (1 + 2 MTP drafts)). Backend order is logged as
`Using ['FLASHINFER_PCIE_IPC', 'CUSTOM', 'PYNCCL']` and `Initialized FlashInfer PCIe IPC ... max_tokens=12`.

| ABBA, engbench | CUSTOM->PYNCCL | PCIe IPC uncapped | PCIe IPC cap 12 |
|---|---|---|---|
| single-stream decode | 118.1 tok/s | +2.9% | **+3.4%** |
| 1 / 2 / 4 streams | 100.9 / 186.9 / 360.9 | +7% / +4.7% / +1.4% | **+7.4% / +4.6% / +2.1%** |
| 8 / 16 / 32 streams | 577 / 833 / 947 | -3.3% / -2.5% / -1.9% | flat (old path) |
| prefill 5k-100k | - | flat | flat |

KV pool unchanged (528,756), GSM8K 97.6% (same as before). FlashInfer tunes once and caches it, so the first restart
takes ~5 min and later ones ~75 s. **After a vLLM upgrade:** re-run the apply script against the new venv (it refuses
to write if the anchors moved), or delete the drop-in; without the patch the cap variable is ignored and the
uncapped backend would lose at high concurrency.

## Skinny GEMV: split-K for the bf16 GDN in_proj_ba (2026-11-01)

nsys of single-stream decode (CUDA profiler hook + nsys capture ranges): ~35% of every step is bf16 work the INT4 recipe
leaves unquantized - the MTP drafter's shared LM head (2 x 1.35 ms GEMV per step, already at memory bandwidth), the MTP
layer's MLP, and the 48 GDN layers' `in_proj_ba` (48 x 5120 per GPU). cuBLAS ran that last one as a 4-block,
one-warp-per-block tile kernel: 29.6 us per call for a 0.5 MB weight, ~1.4 ms per step.

`/opt/vllm/apply_vllm_skinny_splitk.py VENV` (idempotent; backups `utils.py.bak-pre-lab-skinny`, `envs.py.bak-pre-lab-skinny`):
with `VLLM_LAB_SKINNY_SPLITK=1`, unquantized linears with <= 128 output rows, K >= 2048 (K % 256 == 0), no bias go through
a custom op; 2..8-token batches use a Triton split-K GEMV (fp32 accumulation, deterministic, 6.4-7.2 us vs cuBLAS
22.9-27.9 us), every other batch size `F.linear` as before. The flag is registered in `vllm/envs.py`, so it is part of
the torch.compile cache key (a flip recompiles once, ~165 s start).

Enabled by drop-in `vllm-engine.service.d/61-lab-skinny-splitk.conf`. Measured (legs A S S A): decode 32k +4.8%, 100k
+4.6%, short +1.2%; greedy copy edits wall -8.6% (texts identical), chat -4.4%; KV pool unchanged (528,756). Not
bit-identical (sum order): full GSM8K 1266 vs 1267/1319, flips symmetric. Rollback: remove the drop-in,
`systemctl daemon-reload`, restart vllm-engine. The first request after the very first start compiles the Triton kernel
(~0.6 s, cached in ~/.triton afterwards). Lab report: llamacpp-patches/reviews/2026-11-01-0032-midj-and-vllm-splitk/.

Tried and rejected: FP8 MTP draft (`"quantization":"fp8"` in --speculative-config) - vLLM refuses a draft quantization
different from the checkpoint's (compressed-tensors), and the drafter shares the target's LM head anyway.

## Draft head: NVFP4 private copy of the shared lm_head (2026-11-01, vllm#59973 backport)

The MTP drafter shares the target's bf16 lm_head (248320 x 5120, 1.27 GB per GPU) and reads it in full for every draft
token (2 x 1.35 ms GEMV per step, ~15% of decode). `"draft_lm_head_quantization":"nvfp4"` in `--speculative-config`
gives the drafter a private weight-only NVFP4 copy (Marlin W4A16 on sm_86; `fp8` needs FP8 activations = sm_89+). The
target still verifies with its bf16 head, so outputs are unchanged; only acceptance can move. vLLM 0.32 predates the
merge (2026-10-15), so `/opt/vllm/apply_vllm_pr59973_backport.sh` (diff in /root/backports, backups in
/root/backports/pr59973-backup-*) applies it. Measured (legs A D D A): decode +7.8%, 32k +7.2%, edits wall -11.0%,
chat -6.6%, KV pool 528,756 -> 509,393; GSM8K 245/250 after deploy. Rollback: remove the key from vllm-engine.sh
(backup vllm-engine.sh.bak-pre-drafthead-*), restart.

## vLLM main (evaluated 2026-11-01, not deployed)

`/root/vllm-venv-main` = nightly 0.33.0rc1.dev452+g8530a778f (torch 2.14+cu132) + #64999 (restores the GDN in_proj_ba
overlap that #62956 silently disabled) + the PCIe-IPC cap + split-K (hooks `cuda_unquantized_gemm` there). Qwen3.5 on
NVIDIA auto-enables breakable CUDA graphs on main (KV 509k -> 408k, slower): set `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. With
that: decode +11%, 32k +14%, edits -13.6%, TTFT +40 ms, KV 485,562. Still to do before a switch: full GSM8K and a
prefill sweep (held until the internet switch is off this UPS - see below), then update /opt/vllm/engine.expect
(vllm-envcheck pins the venv). Lab report: llamacpp-patches/reviews/2026-11-01-upstream-prs-and-vllm-main/.

## Power: the UPS limit (2026-11-01)

This box's 900 W (950 W peak) battery backup tripped at 22:03 UTC under GSM8K at concurrency 2 on the vLLM-main
variant (2 x 3090 at 380 W + system). The internet switch is on the same UPS, so the whole lab lost network. The user
keeps 380 W and moves the switch to separate power; until then no heavy concurrent-request benchmarks here.

## 2026-11-02: production moved to vLLM main (nightly)
/root/vllm-venv-main = 0.33.0rc1.dev452+g8530a778f, torch 2.14.0+cu132 (+#64999, PCIe-IPC cap, split-K on cuda_unquantized_gemm,
#59973 native), VLLM_USE_BREAKABLE_CUDAGRAPH=0 in vllm-engine.sh. vs 0.32: decode +11%, 32k +14%, 100k +6%, edits -13.6%,
chat -7.6%, prefill flat (5k -2.9%), TTFT +40 ms, KV 485,562. Full GSM8K (c1, breathers) 1264 vs 1268/1319 (4 vs 8 flips).
engine.expect TORCH_VERSION must carry the +cu132 suffix. The gateway (vllm-direct) stays on /root/vllm-venv-032 python.
Rollback: restore /root/backports/vllm-engine.sh.bak-pre-main-* and engine.expect.bak-pre-main-*, restart vllm-engine.

## 2026-11-02: graph capture list cut at 128 (two full 262k slots)
`--compilation-config '{"cudagraph_capture_sizes": [1,2,3,4,6,8,12,16,24,...,128]}'` in vllm-engine.sh: KV 485,562 -> 543,650 (2.07x).
Keep 3/6/12 in the list (12 = 4 streams x 3 MTP = VLLM_PCIE_IPC_MAX_TOKENS). Rollback: /root/backports/vllm-engine.sh.bak-pre-graphlist-*.

## Tiered KV offload / persistent prefix cache (2026-11-10)
`--kv-transfer-config` (OffloadingConnector, TieringOffloadingSpec): 16 GiB CPU tier (/dev/shm region) + NVMe fs tier
`/var/cache/vllm-kvcache` (host dataset Samsung_PM9A1_1TB/vllm-kvcache, quota 200G, atime on) -> prefixes survive engine
restarts. `--max-num-seqs 64` keeps the KV pool at 552,587 (2.1x 262k). Drop-in 62-kvoffload-shm-clean.conf removes a
stale /dev/shm/vllm_offload_*.mmap before each start (a leftover one made the engine crash-loop). The fs tier never deletes:
vllm-kvcache-trim.timer (10 min) evicts least-recently-read chunk files above 90% of the dataset, down to 85%.
Rollback: restore vllm-engine.sh.bak-pre-kvoff-*, remove the drop-in, disable the timer.
