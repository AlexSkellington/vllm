#!/usr/bin/env python3
"""vllm_gateway.py -- llama.cpp-compatible front for a vLLM OpenAI server (lab, 2026-10-28).

Why: every consumer in this lab (CT108 router, CT107 watchdog/monitor/lab map, CT101 MCP model
tools, Open WebUI, OpenCode) was built against llama-server's API: /props, /slots, the `meta` block
in /v1/models, per-frame `timings`, `prompt_progress` frames, `reasoning_content`.  vLLM serves the
OpenAI API but none of those.  This gateway sits on the production port and makes a vLLM engine look
like a direct-mode llama-server, so nothing downstream needs an engine-specific code path.

  client --> :8081 vllm_gateway --> 127.0.0.1:8000 vllm serve (unit vllm-engine.service)

What it adds (everything else is passed through untouched):
  /health            llama.cpp shape; 503 while the engine loads
  /props             n_ctx, total_slots, chat_template (+probed caps), modalities, build_info, engine_unit;
                     every capability is derived from the checkpoint + the engine's live argv/OpenAPI
  /slots             one entry per in-flight request (+ idle slots), with the live counters the
                     router and watchdog read: is_processing, n_prompt_tokens(_processed/_cache),
                     next_token[0].n_decoded, n_ctx, id_task
  /v1/models         llama.cpp `meta` block, context_length, architecture.input_modalities
  /tokenize /detokenize /completion   translated to vLLM's endpoints
  chat/completions   strips llama.cpp-only fields; streams always carry vLLM continuous usage so the
                     gateway can emit llama.cpp `timings` (prompt/cache/predicted/draft) and, on
                     request, `prompt_progress` frames during prefill; delta.reasoning ->
                     reasoning_content; non-stream requests are streamed upstream and re-assembled so
                     they get the same timings and live /slots progress.

Prefill progress: vLLM reports nothing until the first token (its counters update only when a request
finishes - measured).  `n_prompt_tokens_processed` is therefore an ESTIMATE: elapsed x the measured
prefill rate, capped one token short of the total.  Once capped it only advances (+1 per second) while
the GPUs are actually busy and the prefill is younger than GW_STALL_FACTOR x its expected duration --
so a hung engine (idle GPUs, or a prefill far beyond any plausible time) freezes the marker exactly
like a frozen llama-server, and the CT107 watchdog's busy-but-frozen logic works unchanged.
"""
import asyncio, itertools, json, os, re, time, uuid, socket, stat
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("GW_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
LISTEN_HOST = os.environ.get("GW_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("GW_PORT", "8081"))
# Optional override; by default the checkpoint directory is read from the engine's live argv.
MODEL_DIR = Path(os.environ["GW_MODEL_DIR"]) if os.environ.get("GW_MODEL_DIR") else None
TOTAL_SLOTS = int(os.environ.get("GW_TOTAL_SLOTS", "16"))
ENGINE_UNIT = os.environ.get("GW_ENGINE_UNIT", "vllm-engine.service")
PREFILL_RATE0 = float(os.environ.get("GW_PREFILL_RATE", "2000"))      # tok/s before the first measurement
STALL_FACTOR = float(os.environ.get("GW_STALL_FACTOR", "4"))
GPU_BUSY_PCT = float(os.environ.get("GW_GPU_BUSY_PCT", "30"))
PROGRESS_EVERY = float(os.environ.get("GW_PROGRESS_EVERY", "0.5"))

# Request validation (llama.cpp-only fields -> explicit 400, aliases, unknown fields) lives in the shared
# engine_compat module; the accepted vLLM field lists come from the live engine's /openapi.json.
from engine_compat import CompatibilityError, engine_contract, translate_vllm

app = FastAPI()
client = None                       # httpx.AsyncClient to the engine
INFO = {}                           # static model facts, filled at startup
_task_ids = itertools.count(1)
ACTIVE = {}                         # rid -> request record (see _new_record)
# 2026-11-08 parity 3: lab_llama_samplers side channels (see that module)
CONTROL_DIR = "/dev/shm/lab-control"
PSP_DIR = "/dev/shm/lab-psp"
LAB_RIDS = {}                       # lab request id -> reasoning_control armed (live requests only)


class _PspReader:
    """Incremental reader of one request's post-chain distributions: output index -> [(id, prob)...], last write wins
    (a rejected draft row is always rewritten by the later row that produced the committed token)."""

    def __init__(self, rid):
        self.path, self.off, self.t, self.buf = os.path.join(PSP_DIR, rid), 0, {}, b""

    def read(self):
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.off)
                data = fh.read()
        except FileNotFoundError:
            return self.t
        self.off += len(data)
        data = self.buf + data
        cut = data.rfind(b"\n") + 1
        self.buf = data[cut:]
        for line in data[:cut].decode().splitlines():
            i, _, rest = line.partition("\t")
            if rest:
                self.t[int(i)] = [(int(a), float(b)) for a, b in (x.split(":") for x in rest.split(","))]
        return self.t

    def close(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


async def _psp_entries(reader, k0, ids, n):
    """llama.cpp post_sampling_probs entries (id, token, bytes, prob, top_probs) for output indices k0.. ."""
    t = reader.read()
    rows = [t.get(k0 + j, []) for j in range(len(ids))]
    text = await _token_text(sorted({i for i in ids if i >= 0} | {a for r in rows for a, _ in r[:n]}))
    out = []
    for i, r in zip(ids, rows):
        s = text.get(i, "")
        top = [{"id": a, "token": text.get(a, ""), "bytes": list(text.get(a, "").encode("utf-8")), "prob": b} for a, b in r[:n]]
        out.append({"id": i, "token": s, "bytes": list(s.encode("utf-8")), "prob": dict(r).get(i, 0.0), "top_probs": top})
    return out


def _lp_ids(lp):
    """Token ids of a vLLM logprobs object (tokens rendered as 'token_id:N'), chat or completions shape."""
    tid = lambda t: int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else -1
    if isinstance(lp.get("content"), list):
        return [tid(e.get("token")) for e in lp["content"]]
    return [tid(t) for t in lp.get("tokens") or []]


async def _psp_rewrite(obj, reader, n, counter):
    """Replace every choice's logprobs with post-chain probabilities; counter[choice index] = output index reached."""
    for ch in obj.get("choices", []) or []:
        lp = ch.get("logprobs")
        if not isinstance(lp, dict):
            continue
        c = lp.get("content")
        if isinstance(c, list) and c and "prob" in c[0]:
            continue
        ids = _lp_ids(lp)
        if not ids:
            continue
        k0 = counter.get(ch.get("index", 0), 0)
        ch["logprobs"] = {"content": await _psp_entries(reader, k0, ids, n)}
        counter[ch.get("index", 0)] = k0 + len(ids)
STATE = {"prefill_rate": PREFILL_RATE0, "gpu_busy": False, "version": "?", "model_id": None,
         "max_len": None}


# ---------------------------------------------------------------- startup facts / background state
# Everything a model advertises is derived from the checkpoint plus the engine's LIVE argv, the way
# llama-server derives /props from its GGUF, its mmproj and its own template probe: no capability is
# hand-listed, so a different checkpoint or engine flag set advertises exactly what it has.

def _json_file(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _arg(args, flag):
    """Value of a CLI flag in argv: `--flag v` / `--flag=v` -> v, bare switch -> "", absent -> None."""
    for i, a in enumerate(args):
        if a == flag:
            nxt = args[i + 1] if i + 1 < len(args) else ""
            return "" if nxt.startswith("--") else nxt
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _model_dir(args):
    if MODEL_DIR is not None:
        return MODEL_DIR
    if "serve" in args and args.index("serve") + 1 < len(args):
        return Path(args[args.index("serve") + 1])
    model = _arg(args, "--model")
    if model:
        return Path(model)
    raise RuntimeError("cannot determine the checkpoint: engine argv unreadable and GW_MODEL_DIR unset")


def _ftype(cfg, args):
    """Weight/KV precision label from the checkpoint's quantization config and the engine's KV flag."""
    if os.environ.get("GW_FTYPE"):
        return os.environ["GW_FTYPE"]
    q = cfg.get("quantization_config") or {}
    w = {}
    for group in (q.get("config_groups") or {}).values():
        w = group.get("weights") or {}
        break
    if w.get("num_bits"):
        acts = None
        for group in (q.get("config_groups") or {}).values():
            acts = (group.get("input_activations") or {}).get("num_bits")
        weights = "W%sA%s %s%s" % (w["num_bits"], acts or 16, str(w.get("type", "int")).upper(), w["num_bits"])
        if w.get("group_size"):
            weights += " g%s" % w["group_size"]
        weights += " (%s)" % (q.get("quant_method") or q.get("format") or "quantized")
    elif q.get("quant_method"):
        weights = str(q["quant_method"])
    else:
        text = cfg.get("text_config", cfg)
        weights = str(text.get("torch_dtype") or cfg.get("torch_dtype") or cfg.get("dtype") or "unknown")
    kv = _arg(args, "--kv-cache-dtype")
    scheme = q.get("kv_cache_scheme") or {}
    if kv in (None, "", "auto") and scheme.get("num_bits"):
        kv = ("fp%s" if scheme.get("type") == "float" else "int%s") % scheme["num_bits"]
    return "%s, %s KV" % (weights, (kv or "model dtype").upper())


def _mm_limits(args):
    raw = _arg(args, "--limit-mm-per-prompt")
    try:
        limits = json.loads(raw) if raw else {}
    except ValueError:
        limits = {}
    out = {}
    for kind, value in limits.items():
        out[kind] = value.get("count", 1) if isinstance(value, dict) else value
    return out


def _modalities(cfg, model_dir, args):
    """What the engine will really accept: the checkpoint must ship the encoder AND its processor, and
    the engine must not have switched the modality off (--language-model-only, a zero mm limit)."""
    proc = _json_file(model_dir / "processor_config.json")
    if _arg(args, "--language-model-only") is not None:
        return {"vision": False, "video": False, "audio": False}
    limits = _mm_limits(args)
    image_proc = bool(proc.get("image_processor")) or (model_dir / "preprocessor_config.json").is_file()
    video_proc = bool(proc.get("video_processor")) or (model_dir / "video_preprocessor_config.json").is_file()
    audio_proc = bool(proc.get("feature_extractor") or proc.get("audio_processor"))
    vision = bool(cfg.get("vision_config")) and image_proc and limits.get("image", 1) != 0
    return {
        "vision": vision,
        "video": bool(cfg.get("vision_config")) and video_proc and limits.get("video", 1) != 0,
        "audio": bool(cfg.get("audio_config")) and audio_proc and limits.get("audio", 1) != 0,
    }


_NEEDLE = "lab_probe_%s"
# llama-server renders every request with preserve_reasoning on by default (common/arg.cpp), which sets
# these template variables (caps_apply_preserve_reasoning).  The gateway applies the same defaults to
# templates that read them, so a model keeps the same history -- and the same prefix-cache reuse -- on
# either engine.  A client's own chat_template_kwargs still win.
PRESERVE_KWARGS = {"preserve_thinking": True, "clear_thinking": False, "truncate_history_thinking": False,
                   "drop_thinking": False}


def _template_caps(tpl, bos, eos):
    """llama.cpp-equivalent chat_template_caps, measured by rendering the template with probe messages
    through transformers' sandboxed Jinja environment (the renderer vLLM itself uses)."""
    caps = {k: False for k in ("supports_object_arguments", "supports_parallel_tool_calls",
                               "supports_preserve_reasoning", "supports_reasoning_effort",
                               "supports_string_content", "supports_system_role", "supports_tool_calls",
                               "supports_tools", "supports_typed_content")}
    if not tpl:
        return caps, "no chat template"
    try:
        from transformers.utils.chat_template_utils import _compile_jinja_template
        compiled = _compile_jinja_template(tpl)
    except Exception as exc:  # noqa: BLE001
        return caps, "template did not compile: %s" % exc

    def render(messages, tools=None, **kw):
        try:
            return compiled.render(messages=messages, tools=tools, add_generation_prompt=True,
                                   bos_token=bos or "", eos_token=eos or "", **kw)
        except Exception:
            return ""

    user = {"role": "user", "content": "hi"}
    tool = {"type": "function", "function": {
        "name": _NEEDLE % "tool", "description": "probe",
        "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]}}}

    def call(name, arguments):
        return {"id": "call_" + name, "type": "function", "function": {"name": name, "arguments": arguments}}

    def with_calls(calls):
        return [user, {"role": "assistant", "content": "", "tool_calls": calls},
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "ok"}]

    caps["supports_string_content"] = _NEEDLE % "str" in render([{"role": "user", "content": _NEEDLE % "str"}])
    caps["supports_typed_content"] = _NEEDLE % "typed" in render(
        [{"role": "user", "content": [{"type": "text", "text": _NEEDLE % "typed"}]}])
    caps["supports_system_role"] = _NEEDLE % "sys" in render(
        [{"role": "system", "content": _NEEDLE % "sys"}, user])
    caps["supports_tools"] = _NEEDLE % "tool" in render([user], tools=[tool])
    obj = render(with_calls([call(_NEEDLE % "obj", {"arg": _NEEDLE % "objarg"})]), tools=[tool])
    caps["supports_object_arguments"] = _NEEDLE % "obj" in obj and _NEEDLE % "objarg" in obj
    caps["supports_tool_calls"] = caps["supports_object_arguments"] or _NEEDLE % "strcall" in render(
        with_calls([call(_NEEDLE % "strcall", json.dumps({"arg": "x"}))]), tools=[tool])
    args = {"arg": "x"} if caps["supports_object_arguments"] else json.dumps({"arg": "x"})
    par = render(with_calls([call(_NEEDLE % "pa", args), call(_NEEDLE % "pb", args)]), tools=[tool])
    caps["supports_parallel_tool_calls"] = _NEEDLE % "pa" in par and _NEEDLE % "pb" in par
    # Same probe as llama.cpp common/jinja/caps.cpp: reasoning one turn DEEPER than the last assistant
    # message, rendered with thinking on and the preserve switches set (PRESERVE_KWARGS).
    caps["supports_preserve_reasoning"] = _NEEDLE % "think" in render(
        [user, {"role": "assistant", "content": "a", "reasoning_content": _NEEDLE % "think"}, user,
         {"role": "assistant", "content": "b", "reasoning_content": "r"}, user],
        enable_thinking=True, **PRESERVE_KWARGS)
    # A graded effort control changes the rendered prompt; a template that merely mentions the word does not.
    if "reasoning_effort" in tpl:
        variants = {render([user], reasoning_effort=level) for level in ("low", "medium", "high", "xhigh")}
        caps["supports_reasoning_effort"] = len(variants - {""}) > 1
    return caps, "probed"


def _engine_gate(caps, args):
    """Template caps are necessary, not sufficient: vLLM only parses tool calls when it was started with
    --enable-auto-tool-choice and a --tool-call-parser, and only separates reasoning with --reasoning-parser."""
    gated = dict(caps)
    notes = {"tool_call_parser": _arg(args, "--tool-call-parser") or None,
             "reasoning_parser": _arg(args, "--reasoning-parser") or None}
    if _arg(args, "--enable-auto-tool-choice") is None or not notes["tool_call_parser"]:
        for k in ("supports_tools", "supports_tool_calls", "supports_parallel_tool_calls"):
            gated[k] = False
        notes["tools_disabled_by_engine"] = True
    return gated, notes


def _load_static():
    args = _engine_args()
    model_dir = _model_dir(args)
    cfg = json.loads((model_dir / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    gen = _json_file(model_dir / "generation_config.json")
    tok_cfg = _json_file(model_dir / "tokenizer_config.json")
    tpl = ""
    custom = _arg(args, "--chat-template")
    for path in ([Path(custom)] if custom else []) + [model_dir / "chat_template.jinja"]:
        if path.is_file():
            tpl = path.read_text()
            break
    if not tpl:
        tpl = tok_cfg.get("chat_template") or ""
    bos = tok_cfg.get("bos_token") if isinstance(tok_cfg.get("bos_token"), str) else None
    eos = tok_cfg.get("eos_token") if isinstance(tok_cfg.get("eos_token"), str) else None
    caps, caps_source = _template_caps(tpl, bos, eos)
    caps, engine_notes = _engine_gate(caps, args)
    size = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
    n_params = int(os.environ["GW_N_PARAMS"]) if os.environ.get("GW_N_PARAMS") else None
    INFO.update(
        model_dir=model_dir, ftype=_ftype(cfg, args),
        n_vocab=text.get("vocab_size"), n_embd=text.get("hidden_size"),
        n_ctx_train=text.get("max_position_embeddings"), size=size, n_params=n_params,
        modalities=_modalities(cfg, model_dir, args), gen=gen, chat_template=tpl,
        caps=caps, caps_source=caps_source, engine_notes=engine_notes, bos=bos, eos=eos,
        template_defaults={k: v for k, v in PRESERVE_KWARGS.items() if k in tpl},
    )


def _engine_args():
    """The engine's live argv (single source of truth for lab discovery tags), via its systemd MainPID."""
    try:
        import subprocess
        pid = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", ENGINE_UNIT],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if pid and pid != "0":
            return [a for a in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0") if a]
    except Exception:
        pass
    return []


async def _refresh_upstream_facts():
    STATE["engine_args"] = _engine_args() or STATE.get("engine_args", [])
    try:
        r = await client.get(UPSTREAM + "/v1/models", timeout=5)
        row = (r.json().get("data") or [{}])[0]
        STATE["model_id"] = row.get("id")
        STATE["max_len"] = row.get("max_model_len")
        v = await client.get(UPSTREAM + "/version", timeout=5)
        STATE["version"] = v.json().get("version", "?")
    except Exception:
        return False
    if STATE.get("native_version") != STATE["version"]:
        # The request fields this engine accepts, read from its own OpenAPI schema, so a vLLM upgrade
        # changes the contract by itself.  Until it is read, engine_compat's pinned lists apply.
        try:
            o = await client.get(UPSTREAM + "/openapi.json", timeout=10)
            schemas = o.json()["components"]["schemas"]
            STATE["native"] = {"chat": sorted(schemas["ChatCompletionRequest"]["properties"]),
                               "completion": sorted(schemas["CompletionRequest"]["properties"]),
                               "source": "vLLM %s /openapi.json (live)" % STATE["version"]}
            STATE["native_version"] = STATE["version"]
        except Exception:
            pass
    return True


def _gpu_sampler_thread():
    """GPU busy flag for the prefill heartbeat (pynvml, 1 Hz). Never raises."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        STATE["gpu_busy"] = None      # unknown: heartbeat falls back to the time cap alone
        return
    while True:
        try:
            utils = [pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
            STATE["gpu_busy"] = bool(utils) and min(utils) >= GPU_BUSY_PCT
        except Exception:
            STATE["gpu_busy"] = None
        time.sleep(1.0)


async def _heartbeat_loop():
    while True:
        await asyncio.sleep(1.0)
        now = time.monotonic()
        for rec in list(ACTIVE.values()):
            if rec["t_first"] is not None:
                continue
            total = rec["prompt_total"]
            expect = (total or 0) / max(STATE["prefill_rate"], 1.0)
            young = (now - rec["t0"]) < STALL_FACTOR * max(expect, 5.0)
            if STATE["gpu_busy"] is not False and young:
                rec["hb"] += 1


class _PrefillReceiver(asyncio.DatagramProtocol):
    def datagram_received(self, data, addr):
        try:
            if len(data) > 1024:
                return
            item = json.loads(data)
            rid, pos, total, cache, at = (item[k] for k in ("id", "position", "total", "cache", "at"))
            now = time.monotonic()
            if not isinstance(rid, str) or not re.fullmatch(r"[0-9a-f]{32}", rid):
                return
            if any(type(x) is not int for x in (pos, total, cache)) or not (0 <= cache < pos <= total):
                return
            if type(at) not in (int, float) or not (0 <= now - at <= 5) or item.get("preemptions") != 0:
                return
            rec = ACTIVE.get(rid)
            if not rec or rec["t_first"] is not None or at < rec["t0"]:
                return
            previous = rec.get("live_sample")
            if previous and at <= previous["at"]:
                return
            rec["live_sample"] = {"position": pos, "total": total, "cache": cache, "at": at}
            rec["prompt_total"] = total
            rec["cached"] = cache
            rec["cache_known"] = True
            rec["live_rate"] = None
            if previous and previous["total"] == total and previous["cache"] == cache:
                delta, seconds = pos - previous["position"], at - previous["at"]
                if delta > 0 and 0.05 <= seconds <= 10:
                    rec["live_rate"] = {"tokens": delta, "ms": seconds * 1000.0, "at": at}
        except (KeyError, ValueError, TypeError, UnicodeDecodeError, OverflowError):
            return


def _prefill_readable(sock, rec):
    while True:
        try:
            data, _addr = sock.recvfrom(1024)
        except (BlockingIOError, InterruptedError, ConnectionResetError):
            return
        except OSError:
            return
        rec.datagram_received(data, _addr)


async def _start_prefill_receiver():
    path = os.environ.get("GW_PREFILL_SOCKET", "/run/vllm-prefill-progress/progress.sock")
    try:
        folder = Path(path).parent
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if folder.is_symlink() or not folder.is_dir() or folder.stat().st_uid != os.geteuid():
            raise RuntimeError("unsafe prefill telemetry directory")
        os.chmod(folder, 0o700)
        if os.path.lexists(path):
            if not stat.S_ISSOCK(os.lstat(path).st_mode):
                raise RuntimeError("prefill telemetry path is not a socket")
            os.unlink(path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(path)
        os.chmod(path, 0o600)
        sock.setblocking(False)
        asyncio.get_running_loop().add_reader(sock, _prefill_readable, sock, _PrefillReceiver())
        STATE["prefill_socket"] = sock
    except Exception as exc:
        print("vLLM live prefill telemetry unavailable:", type(exc).__name__, flush=True)


@app.on_event("shutdown")
async def _stop_prefill_receiver():
    sock = STATE.pop("prefill_socket", None)
    if sock is not None:
        try:
            asyncio.get_running_loop().remove_reader(sock)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        p = os.environ.get("GW_PREFILL_SOCKET", "/run/vllm-prefill-progress/progress.sock")
        try:
            os.unlink(p)
        except OSError:
            pass


@app.on_event("startup")
async def _startup():
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0),
                               limits=httpx.Limits(max_connections=512, max_keepalive_connections=64))
    _load_static()
    await _refresh_upstream_facts()
    import threading
    threading.Thread(target=_gpu_sampler_thread, daemon=True).start()
    asyncio.ensure_future(_heartbeat_loop())
    await _start_prefill_receiver()


# ---------------------------------------------------------------- request bookkeeping
def _new_record(kind):
    overlap = bool(ACTIVE)
    for previous in ACTIVE.values():
        previous["overlapped"] = True
    used = {r["slot"] for r in ACTIVE.values()}
    slot = next(i for i in itertools.count() if i not in used)
    rid = uuid.uuid4().hex
    rec = {"rid": rid, "kind": kind, "slot": slot, "id_task": next(_task_ids), "t0": time.monotonic(),
           "t_first": None, "t_last": None, "prompt_total": None, "cached": 0, "completion": 0,
           "hb": 0, "spec0": None, "cache_known": False, "engine_metrics": {},
           "prefill_learned": False, "overlapped": overlap}
    ACTIVE[rid] = rec
    return rec


def _est_processed(rec):
    total = rec["prompt_total"]
    if rec["t_first"] is not None:
        return total or 0
    if not total:
        return 0
    est = int((time.monotonic() - rec["t0"]) * STATE["prefill_rate"])
    cap = max(0, total - 1)
    return min(est, cap) + (rec["hb"] if est >= cap else 0)


def _timings(rec, final=False):
    now = time.monotonic()
    metrics = rec.get("engine_metrics") or {}
    pms = metrics.get("time_to_first_token_ms")
    measured_prefill = (rec.get("cache_known") and isinstance(pms, (int, float)) and pms > 0)
    pn = max(0, (rec["prompt_total"] or 0) - rec["cached"]) if measured_prefill else 0
    t = {"cache_n": rec["cached"] if rec.get("cache_known") else 0,
         "cache_known": bool(rec.get("cache_known")), "prompt_n": pn,
         "prompt_ms": pms if measured_prefill else 0.0, "prompt_per_token_ms": 0.0,
         "prompt_per_second": 0.0, "prompt_estimated": not bool(measured_prefill),
         "timing_source": "vllm per-request metrics" if metrics else "vllm gateway stream clock",
         "predicted_n": 0, "predicted_ms": 0.0, "predicted_per_token_ms": 0.0, "predicted_per_second": 0.0}
    if measured_prefill and pn:
        t["prompt_per_token_ms"] = pms / pn
        t["prompt_per_second"] = pn * 1000.0 / pms
    live = rec.get("live_rate")
    if rec["t_first"] is None and live and now - live["at"] <= 3:
        t.update(prompt_n=live["tokens"], prompt_ms=live["ms"],
                 prompt_per_token_ms=live["ms"] / live["tokens"],
                 prompt_per_second=live["tokens"] * 1000.0 / live["ms"],
                 prompt_estimated=False,
                 timing_source="vllm completed scheduler chunks (windowed)")
    if rec["t_first"] is not None:
        t["gateway_ttft_ms"] = (rec["t_first"] - rec["t0"]) * 1000.0
        n = rec["completion"]
        dms = metrics.get("generation_time_ms")
        actual_decode_clock = isinstance(dms, (int, float)) and not isinstance(dms, bool) and __import__("math").isfinite(dms) and dms >= 0
        t["predicted_estimated"] = not actual_decode_clock
        t["predicted_timing_source"] = "vllm per-request generation time" if actual_decode_clock else "vllm gateway stream clock (estimate)"
        if not actual_decode_clock:
            dms = ((rec["t_last"] or now) - rec["t_first"]) * 1000.0
        t.update(predicted_n=n, predicted_ms=round(dms, 3))
        if n > 1 and dms > 0:
            t["predicted_per_token_ms"] = dms / (n - 1)
            t["predicted_per_second"] = (n - 1) * 1000.0 / dms
    if final and rec.get("draft"):
        t.update(rec["draft"])
    return t


async def _spec_counters():
    try:
        txt = (await client.get(UPSTREAM + "/metrics", timeout=2)).text
        d = sum(float(x) for x in re.findall(r"^vllm:spec_decode_num_draft_tokens_total\{[^}]*\} ([0-9.e+]+)", txt, re.M))
        a = sum(float(x) for x in re.findall(r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} ([0-9.e+]+)", txt, re.M))
        return d, a
    except Exception:
        return None


def _learn_prefill_rate(rec):
    if rec.get("prefill_learned") or not rec.get("cache_known") or rec.get("overlapped"):
        return
    pms = (rec.get("engine_metrics") or {}).get("time_to_first_token_ms")
    if not isinstance(pms, (int, float)) or pms <= 0:
        return
    pn = max(0, (rec["prompt_total"] or 0) - rec["cached"])
    rec["prefill_learned"] = True
    if pn >= 2048 and pms > 200:
        # This tunes progress estimation only. Never publish its smoothed rate as measured TPS.
        STATE["prefill_rate"] = 0.7 * STATE["prefill_rate"] + 0.3 * (pn * 1000.0 / pms)


# ---------------------------------------------------------------- simple endpoints
@app.get("/health")
async def health():
    try:
        r = await client.get(UPSTREAM + "/health", timeout=3)
        if r.status_code == 200:
            if not STATE["model_id"]:
                await _refresh_upstream_facts()
            return {"status": "ok"}
    except Exception:
        pass
    return JSONResponse({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}},
                        status_code=503)


def _engine_output_cap():
    """vLLM 0.32 applies override-generation-config.max_new_tokens as a hard cap.

    Read the live engine argv, not a model-name table. A custom generation-config
    path has different resolution rules; leave its limit unknown here.
    """
    args = STATE.get("engine_args") or []
    def option(name, default=None):
        for i, value in enumerate(args):
            if value == name and i + 1 < len(args):
                return args[i + 1]
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
        return default
    if option("--generation-config", "auto") not in ("auto", "vllm"):
        return None
    try:
        overrides = json.loads(option("--override-generation-config", "{}"))
        value = overrides.get("max_new_tokens")
    except (TypeError, ValueError, AttributeError):
        return None
    return value if type(value) is int and value > 0 else None


def _gen_params():
    g = INFO.get("gen") or {}
    n_ctx = STATE["max_len"] or 0
    return {"n_ctx": n_ctx, "temperature": g.get("temperature", 1.0), "top_k": g.get("top_k", 20),
            "top_p": g.get("top_p", 0.95), "min_p": g.get("min_p", 0.0), "presence_penalty": 0.0,
            "repeat_penalty": g.get("repetition_penalty", 1.0), "n_predict": _engine_output_cap() or -1, "seed": -1,
            "stream": False, "speculative.types": _speculative_types()}


def _speculative_types():
    raw = _arg(STATE.get("engine_args", []), "--speculative-config")
    try:
        method = (json.loads(raw) if raw else {}).get("method")
    except (ValueError, AttributeError):
        method = None
    return [method] if method else []


def _contract():
    """The shared contract, filled with this engine's live request schema and the probe results."""
    contract = engine_contract("vllm", native=STATE.get("native"))
    contract["template_caps_source"] = INFO.get("caps_source")
    contract.update(INFO.get("engine_notes") or {})
    return contract


@app.get("/props")
async def props():
    if not STATE["model_id"]:
        await _refresh_upstream_facts()
    n_ctx = STATE["max_len"] or 0
    return {
        "default_generation_settings": {"n_ctx": n_ctx, "params": _gen_params()},
        "total_slots": TOTAL_SLOTS, "model_alias": STATE["model_id"], "model_path": str(INFO.get("model_dir")),
        "model_ftype": INFO.get("ftype"), "modalities": dict(INFO.get("modalities") or {}),
        "chat_template": INFO.get("chat_template", ""),
        "chat_template_caps": dict(INFO.get("caps") or {}),
        "bos_token": INFO.get("bos"), "eos_token": INFO.get("eos"),
        "build_info": "vllm-%s (llama.cpp-compat gateway)" % STATE["version"],
        "generation_limits": {"max_output_tokens": _engine_output_cap()},
        "context_scope": "per_request",
        "compatibility": _contract(),
        "engine": "vllm", "engine_version": STATE["version"], "engine_unit": ENGINE_UNIT,
        "engine_args": STATE.get("engine_args", []),
        "endpoint_slots": True, "endpoint_props": True, "endpoint_metrics": True, "is_sleeping": False,
    }


@app.get("/slots")
async def slots():
    n_ctx = STATE["max_len"] or 0
    by_slot = {r["slot"]: r for r in ACTIVE.values()}
    out = []
    for sid in range(max(TOTAL_SLOTS, max(by_slot, default=-1) + 1)):
        rec = by_slot.get(sid)
        if rec is None:
            out.append({"id": sid, "id_task": -1, "n_ctx": n_ctx, "speculative": True, "is_processing": False,
                        "n_prompt_tokens": 0, "n_prompt_tokens_processed": 0, "n_prompt_tokens_cache": 0,
                        "next_token": [{"has_next_token": False, "has_new_line": False, "n_remain": -1,
                                        "n_decoded": 0}]})
            continue
        out.append({"id": sid, "id_task": rec["id_task"], "n_ctx": n_ctx, "speculative": True,
                    "is_processing": True, "prefill_estimated": rec["t_first"] is None,
                    "engine": "vllm", "n_prompt_tokens": rec["prompt_total"] or 0,
                    "n_prompt_tokens_processed": _est_processed(rec), "n_prompt_tokens_cache": rec["cached"],
                    "next_token": [{"has_next_token": True, "has_new_line": False, "n_remain": -1,
                                    "n_decoded": rec["completion"]}],
                    "params": _gen_params()})
    return out


def _params_from_id(model_id):
    m = re.search(r"(\d+(?:\.\d+)?)B\b", model_id or "")
    return int(float(m.group(1)) * 1e9) if m else None


def _model_row():
    n_ctx = STATE["max_len"]
    m = INFO.get("modalities") or {}
    mods = ["text"] + [name for name, key in (("image", "vision"), ("video", "video"), ("audio", "audio"))
                       if m.get(key)]
    return {"id": STATE["model_id"], "aliases": [], "tags": [], "object": "model", "owned_by": "vllm",
            "created": int(time.time()), "context_length": n_ctx,
            "architecture": {"input_modalities": mods, "output_modalities": ["text"]},
            "meta": {"vocab_type": 2, "n_vocab": INFO.get("n_vocab"), "n_ctx": n_ctx,
                     "n_ctx_train": INFO.get("n_ctx_train"), "n_embd": INFO.get("n_embd"),
                     "n_params": INFO.get("n_params") or _params_from_id(STATE["model_id"]),
                     "size": INFO.get("size"), "ftype": INFO.get("ftype"),
                     "compatibility": _contract(), "chat_template_caps": dict(INFO.get("caps") or {}),
        "engine": "vllm", "engine_version": STATE["version"],
                     "max_output_tokens": _engine_output_cap(), "context_scope": "per_request"}}


@app.get("/v1/models")
@app.get("/models")
async def models():
    if not STATE["model_id"] and not await _refresh_upstream_facts():
        return JSONResponse({"error": {"code": 503, "message": "Loading model"}}, status_code=503)
    return {"object": "list", "data": [_model_row()]}


@app.post("/models/load")
async def models_load(request: Request):
    """llama.cpp router-mode endpoint used by the MCP model switch. A vLLM engine serves exactly one model: loading
    that model is a no-op success (it is resident); anything else is refused the way llama.cpp refuses."""
    b = await request.json()
    if b.get("model") in (STATE["model_id"], None):
        return {"success": True, "model": STATE["model_id"], "note": "single-model vLLM engine; already resident"}
    return JSONResponse({"success": False, "error": "this vLLM engine serves only %s" % STATE["model_id"]},
                        status_code=400)


@app.post("/models/unload")
async def models_unload(request: Request):
    return JSONResponse({"success": False, "error": "single-model vLLM engine: the model cannot be unloaded "
                                                    "(stop vllm-engine.service instead)"}, status_code=400)


@app.post("/tokenize")
async def tokenize(request: Request):
    b = await request.json()
    body = {"model": STATE["model_id"], "add_special_tokens": bool(b.get("add_special", False))}
    if "messages" in b:
        body.update(messages=b["messages"], add_generation_prompt=True)
    else:
        body["prompt"] = b.get("content", "")
    r = await client.post(UPSTREAM + "/tokenize", json=body, timeout=60)
    return {"tokens": r.json().get("tokens", [])}


@app.post("/detokenize")
async def detokenize(request: Request):
    b = await request.json()
    r = await client.post(UPSTREAM + "/detokenize", json={"model": STATE["model_id"], "tokens": b.get("tokens", [])}, timeout=60)
    return {"content": r.json().get("prompt", "")}


@app.get("/metrics")
async def metrics():
    r = await client.get(UPSTREAM + "/metrics", timeout=10)
    return Response(r.content, media_type=r.headers.get("content-type", "text/plain"))


# ---------------------------------------------------------------- generation
def _clean_body(body, chat=True):
    out = translate_vllm(body, STATE["model_id"], chat=chat, native=STATE.get("native"), gen_defaults=INFO.get("gen"))
    defaults = INFO.get("template_defaults") or {}
    if chat and defaults:
        kwargs = body.get("chat_template_kwargs")
        body["chat_template_kwargs"] = {**defaults, **(kwargs if isinstance(kwargs, dict) else {})}
    return out


class _Post:
    """llama.cpp server-side behaviours the gateway reproduces for llama.cpp controls (engine parity v2, 2026-11-07):
    reasoning_format (none / deepseek-legacy reshaping), t_max_predict_ms and n_indent stop rules (server-context.cpp
    process_token), sse_ping_interval comments, and the `compat` report."""

    def __init__(self, post, chat):
        self.p, self.chat = post or {}, chat
        self.rf = self.p.get("reasoning_format")
        self.t_max = float(self.p.get("t_max_predict_ms") or 0) / 1000.0
        self.n_indent = int(self.p.get("n_indent") or 0)
        self.ping = float(self.p.get("sse_ping_interval") or 0)
        self.text = ""              # generated visible text (stop rules)
        self.think_open = False
        self.t_gen = None
        self.stopped = False
        self.held = ""              # n_indent: the current line is held back until its indentation is known

    def compat(self):
        return self.p.get("compat") or {}

    def reshape_delta(self, d):
        """Chat delta: llama.cpp reasoning_format none -> thinking inside content with <think> tags; deepseek-legacy ->
        both. Returns the delta (mutated)."""
        if self.rf not in ("none", "deepseek-legacy") or not isinstance(d, dict):
            return d
        r, c = d.get("reasoning_content"), d.get("content")
        add = ""
        if r:
            if not self.think_open:
                add += "<think>"
                self.think_open = True
            add += r
        if c and self.think_open:
            add += "</think>"
            self.think_open = False
        if add or c:
            d["content"] = add + (c or "")
        if self.rf == "none":
            d.pop("reasoning_content", None)
        return d

    def close_think(self):
        if self.think_open:
            self.think_open = False
            return "</think>"
        return ""

    def filter_text(self, piece):
        """Apply n_indent / t_max_predict_ms to newly generated visible text. Returns (text_to_emit, stop_now)."""
        if self.stopped:
            return "", True
        if self.t_gen is None:
            self.t_gen = time.monotonic()
        out = ""
        for ch in piece:
            if self.n_indent > 0 and "\n" in self.text:
                # inside a line after the first newline: hold until the first non-blank character decides
                self.held += ch
                if ch == "\n":
                    out += self.held
                    self.held = ""
                elif ch not in " \t":
                    line = self.held
                    indent = len(line) - len(line.lstrip(" \t"))
                    if indent < self.n_indent:
                        self.stopped = True       # llama.cpp truncates at the first non-blank char (keeps the indent)
                        out += line[:indent]
                        self.held = ""
                        return out, True
                    out += self.held
                    self.held = ""
            else:
                out += ch
            self.text += ch
        # llama.cpp: checked on a token that contains a newline, against the time since generation started
        if self.t_max > 0 and "\n" in piece and time.monotonic() - self.t_gen > self.t_max:
            self.stopped = True
            return out + self.held, True
        return out, False

    def flush(self):
        h, self.held = self.held, ""
        return h


def _rename_reasoning(obj):
    """vLLM 0.3x streams thinking as `reasoning`; llama.cpp (and our clients) use `reasoning_content`."""
    if isinstance(obj, dict) and "reasoning" in obj:
        val = obj.pop("reasoning")
        if val is not None:
            obj["reasoning_content"] = val


async def _count_prompt(rec, body, path):
    """Exact prompt length for progress/slots, fetched in parallel with the generation."""
    try:
        if path.endswith("/chat/completions"):
            msgs = body.get("messages", [])
            if any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") not in ("text", None)
                                                             for p in m["content"]) for m in msgs if isinstance(m, dict)):
                return                          # media: let the first usage frame report the count (no double preprocessing)
            tb = {"model": STATE["model_id"], "messages": msgs, "add_generation_prompt": True}
            for k in ("tools", "chat_template_kwargs"):
                if body.get(k) is not None:
                    tb[k] = body[k]
        else:
            p = body.get("prompt", "")
            if isinstance(p, list) and p and isinstance(p[0], int):
                rec["prompt_total"] = rec["prompt_total"] or len(p)
                return
            tb = {"model": STATE["model_id"], "prompt": p if isinstance(p, str) else ""}
        r = await client.post(UPSTREAM + "/tokenize", json=tb, timeout=30)
        if rec["prompt_total"] is None:
            rec["prompt_total"] = int(r.json().get("count") or len(r.json().get("tokens", [])))
    except Exception:
        pass


def _absorb_usage(rec, usage):
    if not isinstance(usage, dict):
        return
    if isinstance(usage.get("prompt_tokens"), int):
        rec["prompt_total"] = usage["prompt_tokens"]
    det = usage.get("prompt_tokens_details") or {}
    if isinstance(det.get("cached_tokens"), int):
        rec["cached"] = det["cached_tokens"]
        rec["cache_known"] = True
    if isinstance(usage.get("completion_tokens"), int):
        rec["completion"] = usage["completion_tokens"]


def _progress_frame(rec, chunk_id, created, chat):
    total = rec["prompt_total"] or 0
    live = rec.get("live_sample")
    frame = {"id": chunk_id, "object": "chat.completion.chunk" if chat else "text_completion",
             "created": created, "model": STATE["model_id"],
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]
             if chat else [{"index": 0, "text": "", "finish_reason": None}],
             "timings": _timings(rec),
             "prompt_progress": {"total": total, "cache": rec["cached"], "processed": live["position"] if live else _est_processed(rec),
                                 "estimated": not bool(live), "engine": "vllm",
                                 "time_ms": int((time.monotonic() - rec["t0"]) * 1000)}}
    return "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"


def _prefill_request_headers(path, body, rec):
    # One external request ID must map to exactly one engine sequence. For
    # batched prompts or multiple choices, keep progress unavailable instead
    # of attributing one sequence's work to the entire request.
    if body.get("n", 1) not in (None, 1) or body.get("best_of", 1) not in (None, 1):
        return {}
    if not path.endswith("/chat/completions"):
        prompt = body.get("prompt")
        if not (isinstance(prompt, str) or
                (isinstance(prompt, list) and prompt and all(type(x) is int for x in prompt))):
            return {}
    return {"X-Request-Id": "hlp-" + rec["rid"]}


async def _upstream_lines(path, body, rec, queue):
    """Read the engine's SSE stream into `queue`; None marks the end; an Exception is forwarded."""
    try:
        async with client.stream("POST", UPSTREAM + path, json=body, headers=_prefill_request_headers(path, body, rec)) as r:
            if r.status_code != 200:
                await queue.put(("error", r.status_code, await r.aread()))
                return
            async for line in r.aiter_lines():
                if not line:
                    continue
                # Newer vLLM may report validation failures as SSE error objects
                # after returning HTTP 200. Preserve those for both client modes.
                if line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                        code = payload["error"].get("code")
                        code = code if type(code) is int and 400 <= code <= 599 else 502
                        await queue.put(("error", code, json.dumps(payload).encode("utf-8")))
                        return
                await queue.put(("line", line))
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        await queue.put(("exc", repr(e)))
    finally:
        await queue.put(None)


async def _generate(request: Request, path: str):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"code": 400, "message": "invalid JSON"}}, status_code=400)
    chat = path.endswith("/chat/completions")
    original = dict(body)
    try:
        want_timings, want_progress, post = _clean_body(body, chat=chat)
    except CompatibilityError as exc:
        return JSONResponse({"error": {"type": "invalid_request_error", "message": str(exc)}}, status_code=400)
    pp = _Post(post, chat)
    lab_rid = post.get("lab_rid")
    psp_n = post.get("psp") or 0
    psp = _PspReader(lab_rid) if (lab_rid and psp_n) else None
    if lab_rid:
        LAB_RIDS[lab_rid] = bool(post.get("reasoning_control"))

    def lab_done():
        if lab_rid:
            LAB_RIDS.pop(lab_rid, None)
            try:
                os.remove(os.path.join(CONTROL_DIR, lab_rid))
            except OSError:
                pass
        if psp:
            psp.close()
    client_stream = bool(body.get("stream"))
    so = body.get("stream_options") or {}
    client_usage = bool(so.get("include_usage")) if client_stream else True
    # Re-assembling a non-stream reply is only done for the plain shapes; anything exotic passes through.
    passthrough = (not client_stream) and (body.get("n", 1) != 1 or body.get("logprobs") or body.get("echo"))
    rec = _new_record("chat" if chat else "completion")
    if passthrough:
        try:
            r = await client.post(UPSTREAM + path, json=body)
            data = r.json()
            for ch in data.get("choices", []) if isinstance(data, dict) else []:
                _rename_reasoning(ch.get("message"))
                m = ch.get("message")
                if pp.rf in ("none", "deepseek-legacy") and isinstance(m, dict) and m.get("reasoning_content"):
                    m["content"] = "<think>" + m["reasoning_content"] + "</think>" + (m.get("content") or "")
                    if pp.rf == "none":
                        m.pop("reasoning_content", None)
            if isinstance(data, dict) and r.status_code == 200:
                if lab_rid:
                    data["id"] = ("chatcmpl-" if chat else "cmpl-") + lab_rid
                if psp:
                    await _psp_rewrite(data, psp, psp_n, {})
                if pp.compat():
                    data["compat"] = pp.compat()
                if pp.p.get("verbose"):
                    data["__verbose"] = {"request": original, "translated": body}
            return JSONResponse(data, status_code=r.status_code)
        finally:
            ACTIVE.pop(rec["rid"], None)
            lab_done()
    body["stream"] = True
    body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
    asyncio.ensure_future(_count_prompt(rec, body, path))
    rec["spec0"] = await _spec_counters()
    queue = asyncio.Queue()
    reader = asyncio.ensure_future(_upstream_lines(path, body, rec, queue))
    chunk_id = ("chatcmpl-" if chat else "cmpl-") + (lab_rid or uuid.uuid4().hex[:24])
    psp_counter = {}
    created = int(time.time())

    async def finish_stats():
        if len([r for r in ACTIVE.values() if r is not rec]) == 0 and rec["spec0"]:
            now = await _spec_counters()
            if now:
                d, a = int(now[0] - rec["spec0"][0]), int(now[1] - rec["spec0"][1])
                if d > 0:
                    rec["draft"] = {"draft_n": d, "draft_n_accepted": a}

    async def events():
        """Yields ('frame', dict) for each engine chunk, ('progress', None) ticks, ('error', ...)."""
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=PROGRESS_EVERY)
            except asyncio.TimeoutError:
                if rec["t_first"] is None:
                    yield ("progress", None)
                continue
            if item is None:
                return
            if item[0] != "line":
                yield item
                continue
            line = item[1]
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                frame = json.loads(payload)
            except ValueError:
                continue
            yield ("frame", frame)

    def shape(frame, final=False):
        _absorb_usage(rec, frame.get("usage"))
        if isinstance(frame.get("metrics"), dict):
            rec["engine_metrics"].update(frame["metrics"])
        has_output = False
        for ch in frame.get("choices", []) or []:
            _rename_reasoning(ch.get("delta"))
            for k in ("token_ids",):
                if ch.get(k) is None:
                    ch.pop(k, None)
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("tool_calls") or ch.get("text"):
                has_output = True
            if ch.get("finish_reason"):
                has_output = True
        for k in ("prompt_token_ids", "prompt_text"):
            if frame.get(k) is None:
                frame.pop(k, None)
        now = time.monotonic()
        if (has_output or rec["completion"] > 0) and rec["t_first"] is None:
            rec["t_first"] = now
            # Final engine timing/cache statistics, not first-frame guesses, train progress.
        if has_output:
            rec["t_last"] = now
        _learn_prefill_rate(rec)
        return frame

    # Validation errors come back at once; surface them as a real HTTP error, not a 200 stream.
    try:
        first = await asyncio.wait_for(queue.get(), timeout=0.3)
    except asyncio.TimeoutError:
        first = "pending"
    if isinstance(first, tuple) and first[0] in ("error", "exc"):
        reader.cancel()
        ACTIVE.pop(rec["rid"], None)
        if first[0] == "exc":
            return JSONResponse({"error": {"code": 502, "message": first[1]}}, status_code=502)
        try:
            return JSONResponse(json.loads(first[2]), status_code=first[1])
        except Exception:
            return JSONResponse({"error": {"code": first[1], "message": first[2][:2000].decode(errors="replace")}},
                                status_code=first[1])
    if first != "pending":
        queue._queue.appendleft(first)          # put the peeked item back at the head

    def apply_post(frame, finishing):
        """llama.cpp-control behaviours on one engine frame. Returns (frame, stop_now)."""
        stop = False
        for ch in frame.get("choices", []) or []:
            if chat:
                d = pp.reshape_delta(ch.get("delta"))
                if isinstance(d, dict) and d.get("content"):
                    d["content"], stop = pp.filter_text(d["content"])
                if (finishing or stop) and isinstance(d, dict):
                    tail = pp.close_think() + pp.flush()
                    if tail:
                        d["content"] = (d.get("content") or "") + tail
            elif ch.get("text"):
                ch["text"], stop = pp.filter_text(ch["text"])
                if finishing or stop:
                    ch["text"] += pp.flush()
            if stop:
                ch["finish_reason"] = "length"         # llama.cpp STOP_TYPE_LIMIT
        if (finishing or stop) and pp.compat():
            frame["compat"] = pp.compat()
        return frame, stop

    if client_stream:
        async def sse():
            last_emit = time.monotonic()
            try:
                async for kind, *rest in events():
                    if kind == "progress":
                        if want_progress:
                            yield _progress_frame(rec, chunk_id, created, chat)
                            last_emit = time.monotonic()
                        elif pp.ping > 0 and time.monotonic() - last_emit >= pp.ping:
                            yield ": ping\n\n"                # llama.cpp sse_ping_interval comment
                            last_emit = time.monotonic()
                        continue
                    if kind in ("error", "exc"):
                        msg = rest[1].decode(errors="replace") if kind == "error" else rest[0]
                        yield "data: " + json.dumps({"error": {"code": rest[0] if kind == "error" else 500,
                                                               "message": msg[:2000]}}) + "\n\n"
                        return
                    frame = shape(rest[0])
                    if lab_rid and "id" in frame:
                        frame["id"] = chunk_id               # the id the control endpoint knows
                    if psp:
                        await _psp_rewrite(frame, psp, psp_n, psp_counter)
                    is_usage_only = not frame.get("choices")
                    finishing = is_usage_only or any(c.get("finish_reason") for c in frame.get("choices", []))
                    frame, stop_now = apply_post(frame, finishing)
                    finishing = finishing or stop_now
                    if finishing:
                        await finish_stats()
                    if is_usage_only and not client_usage:
                        continue                      # client did not ask for usage
                    if not is_usage_only:
                        frame.pop("usage", None)      # OpenAI shape: usage only on the final usage frame
                    if want_timings or finishing:
                        frame["timings"] = _timings(rec, final=finishing)
                    yield "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"
                    last_emit = time.monotonic()
                    if stop_now:
                        break                          # stop rule fired: end the stream (reader is cancelled below)
                yield "data: [DONE]\n\n"
            finally:
                reader.cancel()
                ACTIVE.pop(rec["rid"], None)
                lab_done()
        return StreamingResponse(sse(), media_type="text/event-stream")

    # Non-stream client: stream upstream, re-assemble.
    try:
        content, reasoning, text, tools, finish, usage, role = [], [], [], {}, None, None, "assistant"
        async for kind, *rest in events():
            if kind == "progress":
                continue
            if kind == "error":
                try:
                    return JSONResponse(json.loads(rest[1]), status_code=rest[0])
                except Exception:
                    return JSONResponse({"error": {"code": rest[0], "message": rest[1][:2000].decode(errors="replace")}},
                                        status_code=rest[0])
            if kind == "exc":
                return JSONResponse({"error": {"code": 502, "message": rest[0]}}, status_code=502)
            frame = shape(rest[0])
            fin_frame = any(c.get("finish_reason") for c in frame.get("choices", []) or [])
            frame, stop_now = apply_post(frame, fin_frame)
            if frame.get("usage"):
                usage = frame["usage"]
            for ch in frame.get("choices", []) or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if chat:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(d["content"])
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"])
                    for tc in d.get("tool_calls") or []:
                        slot = tools.setdefault(tc.get("index", 0), {"id": None, "type": "function",
                                                                     "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                else:
                    if ch.get("text"):
                        text.append(ch["text"])
            if stop_now:
                break                                  # llama.cpp stop rule (n_indent / t_max_predict_ms)
        await finish_stats()
        if chat:
            msg = {"role": role, "content": "".join(content) if content or not tools else None}
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning)
            if tools:
                msg["tool_calls"] = [tools[i] for i in sorted(tools)]
            choice = {"index": 0, "message": msg, "finish_reason": finish, "logprobs": None}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": "".join(text), "finish_reason": finish, "logprobs": None}
            obj = "text_completion"
        out = {"id": chunk_id, "object": obj, "created": created, "model": STATE["model_id"],
               "choices": [choice], "usage": usage, "timings": _timings(rec, final=True)}
        if pp.compat():
            out["compat"] = pp.compat()
        if pp.p.get("verbose"):
            out["__verbose"] = {"request": original, "translated": body}
        if pp.p.get("return_tokens") or pp.p.get("response_fields"):
            out["_post"] = {k: pp.p[k] for k in ("return_tokens", "response_fields") if pp.p.get(k)}
        return out
    finally:
        reader.cancel()
        ACTIVE.pop(rec["rid"], None)
        lab_done()


@app.post("/v1/chat/completions/control")
async def chat_completions_control(request: Request):
    """llama.cpp POST /v1/chat/completions/control {id, action: "reasoning_end"}: end the live completion's reasoning
    now (the processor forces the budget message + end tag, agreed on every tensor-parallel rank)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    cid = body.get("id") or ""
    if not cid:
        return JSONResponse({"error": {"code": 400, "type": "invalid_request_error", "message": "missing completion id"}}, status_code=400)
    if body.get("action") != "reasoning_end":
        return JSONResponse({"error": {"code": 400, "type": "invalid_request_error", "message": "unknown control action"}}, status_code=400)
    rid = cid.split("-", 1)[1] if cid.startswith(("chatcmpl-", "cmpl-")) else cid
    if rid not in LAB_RIDS:
        return {"success": False, "message": "no active completion for this id"}
    if not LAB_RIDS[rid]:
        return {"success": False, "message": "reasoning control not enabled for this completion"}
    os.makedirs(CONTROL_DIR, exist_ok=True)
    with open(os.path.join(CONTROL_DIR, rid), "w") as fh:
        fh.write("reasoning_end")
    return {"success": True}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    return await _generate(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _generate(request, "/v1/completions")


@app.post("/completion")
@app.post("/completions")
async def llama_completion(request: Request):
    """llama.cpp-native /completion -> /v1/completions (non-stream and stream), answered in llama.cpp's shape:
    content/stop/stop_type, tokens (return_tokens), completion_probabilities (n_probs), response_fields, compat."""
    b = await request.json()
    if not isinstance(b, dict):
        return JSONResponse({"error": {"type": "invalid_request_error", "message": "Request body must be a JSON object"}}, status_code=400)
    # Preserve every supplied control for shared validation; never pre-filter knobs.
    body = dict(b)
    body.setdefault("prompt", "")
    n_probs = body.get("n_probs") or 0
    if n_probs:
        body["return_tokens_as_token_ids"] = True     # ids for completion_probabilities; text via _token_text
    if n_probs or body.get("return_tokens"):
        body["return_token_ids"] = True
    fields = body.get("response_fields")
    want_tokens = bool(body.get("return_tokens"))       # translate_vllm consumes the key; remember it here
    request._json = body                     # reuse the shared path with the translated body
    resp = await _generate(request, "/v1/completions")
    if isinstance(resp, StreamingResponse):
        async def native():
            async for chunk in resp.body_iterator:
                s = chunk if isinstance(chunk, str) else chunk.decode()
                if s.startswith(": "):
                    yield s                                   # sse ping comment
                    continue
                if not s.startswith("data: ") or s.startswith("data: [DONE]"):
                    continue
                f = json.loads(s[6:])
                if "error" in f:
                    yield "data: " + json.dumps(f) + "\n\n"
                    continue
                ch = (f.get("choices") or [{}])[0]
                out = {"content": ch.get("text", ""), "stop": bool(ch.get("finish_reason"))}
                if ch.get("token_ids") and want_tokens:
                    out["tokens"] = ch["token_ids"]
                if n_probs and ch.get("logprobs"):
                    out["completion_probabilities"] = await _llama_probs(ch["logprobs"])
                for k in ("timings", "prompt_progress", "compat"):
                    if k in f:
                        out[k] = f[k]
                if out["stop"]:
                    out["stop_type"] = "eos" if ch.get("finish_reason") == "stop" else "limit"
                yield "data: " + json.dumps(out) + "\n\n"
        return StreamingResponse(native(), media_type="text/event-stream")
    if isinstance(resp, JSONResponse):                 # passthrough path (n_probs / n / echo) or an error
        try:
            data = json.loads(bytes(resp.body))
        except Exception:
            return resp
        if resp.status_code != 200 or not isinstance(data, dict) or "choices" not in data:
            return resp
        resp = data
    if isinstance(resp, dict):
        ch = resp["choices"][0]
        u = resp.get("usage") or {}
        out = {"content": ch.get("text", ""), "stop": True, "model": resp["model"],
               "stop_type": "eos" if ch.get("finish_reason") == "stop" else "limit",
               "tokens_predicted": u.get("completion_tokens"), "tokens_evaluated": u.get("prompt_tokens"),
               "timings": resp.get("timings")}
        if want_tokens:
            out["tokens"] = ch.get("token_ids") or []
        if n_probs and ch.get("logprobs"):
            out["completion_probabilities"] = await _llama_probs(ch["logprobs"])
        for k in ("compat", "__verbose"):
            if k in resp:
                out[k] = resp[k]
        if fields:
            out = _project(out, fields)
        return out
    return resp


_TOKEN_TEXT = {}


async def _token_text(ids):
    """Token id -> (text, bytes), cached; vLLM /detokenize per unseen id."""
    missing = [i for i in set(ids) if i not in _TOKEN_TEXT]

    async def one(i):
        try:
            r = await client.post(UPSTREAM + "/detokenize", json={"model": STATE["model_id"], "tokens": [i]}, timeout=30)
            _TOKEN_TEXT[i] = r.json().get("prompt", "")
        except Exception:
            _TOKEN_TEXT[i] = ""
    for k in range(0, len(missing), 64):
        await asyncio.gather(*(one(i) for i in missing[k:k + 64]))
    return {i: _TOKEN_TEXT[i] for i in ids}


async def _llama_probs(lp):
    """vLLM completions logprobs (tokens as 'token_id:N') -> llama.cpp completion_probabilities."""
    if isinstance(lp.get("content"), list) and lp["content"] and "prob" in lp["content"][0]:
        return lp["content"]                     # post_sampling_probs: already llama.cpp entries (_psp_rewrite)
    toks = lp.get("tokens") or []
    tops = lp.get("top_logprobs") or [None] * len(toks)
    tid = lambda t: int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else -1
    ids = [tid(t) for t in toks] + [tid(t) for d in tops if d for t in d]
    text = await _token_text([i for i in ids if i >= 0])
    out = []
    for t, l, d in zip(toks, lp.get("token_logprobs") or [], tops):
        i = tid(t)
        s = text.get(i, "")
        row = {"id": i, "token": s, "logprob": l, "bytes": list(s.encode("utf-8")), "top_logprobs": []}
        for tt, ll in sorted((d or {}).items(), key=lambda kv: -kv[1]):
            j = tid(tt)
            sj = text.get(j, "")
            row["top_logprobs"].append({"id": j, "token": sj, "logprob": ll, "bytes": list(sj.encode("utf-8"))})
        out.append(row)
    return out


def _project(obj, fields):
    """llama.cpp response_fields: keep only the listed fields; 'a/b' paths are flattened to key 'a/b'."""
    res = {}
    for f in fields:
        cur, ok = obj, True
        for part in str(f).split("/"):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            res[f] = cur
    return res


# ---------------------------------------------------------------- everything else: pass through
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(path: str, request: Request):
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
    r = await client.request(request.method, UPSTREAM + "/" + path, content=raw, headers=headers,
                             params=dict(request.query_params), timeout=600)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


if __name__ == "__main__":
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="warning", timeout_keep_alive=30)
