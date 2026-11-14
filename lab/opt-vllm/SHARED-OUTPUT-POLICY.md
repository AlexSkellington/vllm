# One shared output ceiling

The only editable ceiling is **Ai-Hub CT108 `/opt/model-control/generation-policy.json`**, key **`max_output_tokens`**. Current value: **16,384**. This covers generated text and reasoning in each response. Smaller caller requests remain smaller.

Use this command from this Windows machine to change that single setting, reconcile AS1 when needed, and refresh OpenCode:

```powershell
python "C:\Users\Alex_C.T\Nextcloud\Alex_C.T\AI-Lab-Audit-2026-10\bench\shared-output-policy-2026-10-30\set-output-limit.py" 16384
```

Replace the final number with the desired ceiling. Omitting it reapplies the saved policy. The command uses existing HomeLab credentials without printing them. Private client config backups go under `.agents/backups/`.

The router hot-reads the file and applies it to both llama.cpp and vLLM generation routes, including OpenAI aliases and native completion requests. Both listeners expose `/_router/generation-policy` and advertise effective limits in `/v1/models`. OpenCode's existing generator follows that metadata; do not manually edit generated model limits.

AS1's engine startup now derives `--override-generation-config` from this same policy through `/opt/vllm/shared_generation_policy.py`. A generated last-good cache allows startup while the router is temporarily unavailable; the cache is not an editable setting. The engine's current ceiling already matched16,384, so this deployment did not restart it.

Changing the engine ceiling later requires an engine restart. The apply command checks for active requests, refuses to restart a busy engine, and waits for health after a required restart. A warm-up can take several minutes. If an apply fails after saving a policy, rerun without a number during an idle window; the saved file remains the desired setting. The future changed-ceiling restart branch was not exercised against the production GPUs in this task; the idempotent current-policy apply was tested live.

Model/client constraints can remain lower than the shared ceiling: housekeeping keeps its existing8,192-token client budget, and this OpenCode build itself caps output requests at32,000. Both Qwen backends and their OpenCode profiles now use16,384.

Direct raw backend endpoints used for maintenance/benchmarks remain direct. The cross-engine enforcement point is the shared router; raw Pascal server `/props` still reports its native `n_predict=-1`. Regular OpenCode and Open WebUI traffic uses the shared router. This is a serving policy, not a network access-control change.

Snapshots of the JSON, old sources, derived client configs and historical ledgers are evidence, not additional settings. Legacy `ROUTER_OUTPUT_FRACTION`/`ROUTER_OUTPUT_MAX` are no longer consumed; the old agents output-budget drop-in was retired with a backup.
