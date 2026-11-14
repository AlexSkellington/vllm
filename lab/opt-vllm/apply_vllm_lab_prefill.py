#!/usr/bin/env python3
"""apply_vllm_lab_prefill.py VENV (2026-11-07, Claude; carries Codex's 11-05 hook): install the best-effort prefill-progress
telemetry (vllm/v1/core/sched/lab_prefill.py, copied from the production venv) into another vLLM venv and call it from
Scheduler.update_from_output right after the drop-mode stale-output skip. Never affects inference. Idempotent; backup
scheduler.py.bak-pre-lab-prefill."""
import shutil, sys
from pathlib import Path
SRC = Path("/root/vllm-venv-main/lib/python3.12/site-packages/vllm/v1/core/sched/lab_prefill.py")
d = Path(sys.argv[1]) / "lib/python3.12/site-packages/vllm/v1/core/sched"
s = (d / "scheduler.py").read_text()
if "lab_prefill_completed" in s:
    print("already applied"); sys.exit(0)
imp = "from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs\n"
call = ("            # Drop-mode stale output (same-step resume) is discarded entirely.\n"
        "            if output_is_stale and request.drop_stale_output:\n"
        "                continue\n")
assert s.count(imp) == 1 and s.count(call) == 1, "anchors moved"
s = s.replace(imp, imp + "from vllm.v1.core.sched.lab_prefill import completed as lab_prefill_completed\n")
s = s.replace(call, call + "\n            lab_prefill_completed(request, output_is_stale)\n")
shutil.copy2(d / "scheduler.py", d / "scheduler.py.bak-pre-lab-prefill")
shutil.copy2(SRC, d / "lab_prefill.py")
(d / "scheduler.py").write_text(s)
print("applied:", d / "scheduler.py")
