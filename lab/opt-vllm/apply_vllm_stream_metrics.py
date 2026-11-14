#!/usr/bin/env python3
"""apply_vllm_stream_metrics.py VENV -- lab-local vLLM patch (2026-11-08): live per-request timing on every stream chunk.

With --enable-per-request-metrics vLLM attaches {time_to_first_token_ms, generation_time_ms, mean_itl_ms, ...} only to
the FINAL usage chunk of a stream. Mid-stream the lab gateway therefore had no engine clock and timed decode with its
own stream clock, which the footers show as an estimate ("Gen ≈149.1 tok/s"). This patch attaches the same
build_per_request_timing_metrics(res.metrics, completion_tokens) -- vLLM's own function, from the engine's own
first/last-token timestamps -- to each chunk that already carries continuous usage stats (stream_options
continuous_usage_stats, which the gateway always requests). Chat and completions. Final chunk unchanged.
Idempotent per file; refuses to write if an anchor moved; never writes through a hardlink."""
import os, sys
from pathlib import Path

venv = Path(sys.argv[1])
MARK = "lab patch: live per-request timing metrics"
ADD = ("                        if self.enable_per_request_metrics and (request.n or 1) == 1:  # %s\n"
       "                            chunk.metrics = build_per_request_timing_metrics(res.metrics, completion_tokens)\n" % MARK)
TARGETS = {
    "entrypoints/openai/chat_completion/serving.py": (
        "                    if include_continuous_usage:\n"
        "                        completion_tokens = processor.num_output_tokens\n"
        "                        chunk.usage = UsageInfo(\n"
        "                            prompt_tokens=num_prompt_tokens,\n"
        "                            completion_tokens=completion_tokens,\n"
        "                            total_tokens=num_prompt_tokens + completion_tokens,\n"
        "                            completion_tokens_details=(\n"
        "                                _make_completion_tokens_details(\n"
        "                                    previous_reasoning_tokens[i]\n"
        "                                )\n"
        "                                if self._include_reasoning_tokens_details\n"
        "                                else None\n"
        "                            ),\n"
        "                        )\n"),
    "entrypoints/openai/completion/serving.py": (
        "                    if include_continuous_usage:\n"
        "                        prompt_tokens = num_prompt_tokens[prompt_idx]\n"
        "                        completion_tokens = previous_num_tokens[i]\n"
        "                        chunk.usage = UsageInfo(\n"
        "                            prompt_tokens=prompt_tokens,\n"
        "                            completion_tokens=completion_tokens,\n"
        "                            total_tokens=prompt_tokens + completion_tokens,\n"
        "                        )\n"),
}
base = next(venv.glob("lib/python3*/site-packages/vllm"))
for rel, anchor in TARGETS.items():
    f = base / rel
    s = f.read_text()
    if MARK in s:
        print("already applied:", f)
        continue
    assert s.count(anchor) == 1, "anchor moved in %s (%d matches)" % (f, s.count(anchor))
    assert "build_per_request_timing_metrics" in s, "import missing in %s" % f
    backup = f.with_name(f.name + ".bak-pre-lab-stream-metrics")
    if not backup.exists():
        backup.write_text(s)
    s = s.replace(anchor, anchor + ADD)
    os.unlink(f)  # break any hardlink (venv copies made with cp -al share inodes)
    f.write_text(s)
    print("patched", f)
