"""Lab inference compatibility contract; shared byte-for-byte by router (CT108) and gateway (AS1).

v2 (2026-11-07, Claude; user: "support natively both engines completely", "completely parity"): llama.cpp's request
controls are IMPLEMENTED on vLLM instead of rejected. Rules (reviews/2026-11-07-engine-parity/PARITY-MATRIX.md):
  1. a name both engines share keeps the serving engine's meaning;
  2. a request using any llama.cpp-only sampler control is "llama.cpp dialect": its whole sampler chain runs as
     llama.cpp would, in the vLLM logits processor lab_llama_samplers.LlamaSamplerChain (vllm_xargs.llama), with vLLM's
     own temperature/top-k/top-p/min-p/penalties neutralised;
  3. nothing is dropped silently: controls with no object in this deployment are reported in `compat.noop`, and only a
     control that cannot be implemented is a 400.
"""
import uuid


class CompatibilityError(Exception):
    pass


class CapabilityUnavailable(CompatibilityError):
    status_code = 503


VLLM_CHAT_NATIVE = frozenset(['add_generation_prompt', 'add_special_tokens', 'allowed_token_ids', 'bad_words', 'cache_salt', 'chat_template', 'chat_template_kwargs', 'continue_final_message', 'documents', 'ec_transfer_params', 'echo', 'frequency_penalty', 'ignore_eos', 'include_reasoning', 'include_stop_str_in_output', 'kv_transfer_params', 'length_penalty', 'logit_bias', 'logprob_token_ids', 'logprobs', 'max_completion_tokens', 'max_tokens', 'media_io_kwargs', 'messages', 'min_p', 'min_tokens', 'mm_payload', 'mm_processor_kwargs', 'model', 'n', 'parallel_tool_calls', 'presence_penalty', 'priority', 'prompt_logprobs', 'reasoning_effort', 'repetition_detection', 'repetition_penalty', 'request_id', 'response_format', 'return_mm_kwargs', 'return_prompt_text', 'return_token_ids', 'return_token_offsets', 'return_tokens_as_token_ids', 'routed_experts_prompt_start', 'seed', 'session_id', 'skip_special_tokens', 'spaces_between_special_tokens', 'stop', 'stop_token_ids', 'stream', 'stream_interval', 'stream_options', 'structured_outputs', 'temperature', 'thinking_token_budget', 'tool_choice', 'tools', 'top_k', 'top_logprobs', 'top_p', 'truncate_prompt_tokens', 'truncation_side', 'use_beam_search', 'user', 'vllm_xargs', 'watermarking'])
VLLM_COMPLETION_NATIVE = frozenset(['add_special_tokens', 'allowed_token_ids', 'bad_words', 'cache_salt', 'ec_transfer_params', 'echo', 'frequency_penalty', 'ignore_eos', 'include_stop_str_in_output', 'kv_transfer_params', 'length_penalty', 'logit_bias', 'logprob_token_ids', 'logprobs', 'max_tokens', 'min_p', 'min_tokens', 'mm_payload', 'model', 'n', 'presence_penalty', 'priority', 'prompt', 'prompt_embeds', 'prompt_logprobs', 'repetition_detection', 'repetition_penalty', 'request_id', 'response_format', 'return_mm_kwargs', 'return_token_ids', 'return_token_offsets', 'return_tokens_as_token_ids', 'routed_experts_prompt_start', 'seed', 'session_id', 'skip_special_tokens', 'spaces_between_special_tokens', 'stop', 'stop_token_ids', 'stream', 'stream_interval', 'stream_options', 'structured_outputs', 'suffix', 'temperature', 'thinking_token_budget', 'top_k', 'top_p', 'truncate_prompt_tokens', 'truncation_side', 'use_beam_search', 'user', 'vllm_xargs', 'watermarking'])

# llama.cpp server-schema.cpp (9f0e509) sampler controls that select llama.cpp dialect on vLLM
LLAMA_SAMPLER = frozenset(['repeat_penalty', 'repeat_last_n', 'dry_multiplier', 'dry_base', 'dry_allowed_length',
                           'dry_penalty_last_n', 'dry_sequence_breakers', 'top_n_sigma', 'typical_p', 'xtc_probability',
                           'xtc_threshold', 'min_keep', 'dynatemp_range', 'dynatemp_exponent', 'mirostat', 'mirostat_tau',
                           'mirostat_eta', 'adaptive_target', 'adaptive_decay', 'samplers'])
# shared names: llama.cpp meaning inside a dialect request (they move into the chain), vLLM meaning otherwise
SHARED_SAMPLER = ('temperature', 'top_k', 'top_p', 'min_p', 'frequency_penalty', 'presence_penalty')
# llama.cpp controls the gateway implements outside the sampler
LLAMA_GATEWAY = frozenset(['n_predict', 'n_cmpl', 'n_probs', 'cache_prompt', 'grammar', 'json_schema', 'reasoning_format',
                           'reasoning_in_content', 'reasoning_budget_tokens', 'thinking_budget_tokens', 'return_tokens',
                           'timings_per_token', 'return_progress', 'response_fields', 't_max_predict_ms', 'n_indent',
                           'sse_ping_interval', 'verbose', 'lora', 'post_sampling_probs',
                           # 2026-11-08 parity 3: run in lab_llama_samplers (budget message / runtime control / post-chain probs)
                           'reasoning_budget_message', 'reasoning_control', 'grammar_lazy', 'grammar_triggers'])
# llama.cpp controls that need the lab processor even without sampler knobs
PROCESSOR_CONTROLS = ('reasoning_budget_message', 'reasoning_control', 'post_sampling_probs')
# llama.cpp controls with no object on this vLLM deployment (reported, never silently dropped)
LLAMA_NOOP = {
    'id_slot': 'vLLM has no slots; its prefix cache is global, so slot affinity has nothing to pin',
    'slot_id': 'vLLM has no slots; its prefix cache is global, so slot affinity has nothing to pin',
    'n_cache_reuse': 'KV-shift reuse: vLLM reuses prefixes by block hash on every request',
    'cache_reuse': 'KV-shift reuse: vLLM reuses prefixes by block hash on every request',
    'backend_sampling': 'vLLM always samples on the GPU',
    'preserved_tokens': 'tool-call tokenizer aid; vLLM\'s tool parser handles the same tokens',
    'n_keep': 'applies only to llama.cpp context shifting; vLLM rejects an over-long prompt instead of shifting',
    'n_discard': 'applies only to llama.cpp context shifting; vLLM rejects an over-long prompt instead of shifting',
    'speculative.n_max': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
    'speculative.n_min': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
    'speculative.p_min': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
    'speculative.type': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
    'speculative.ngram_size_n': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
    'speculative.ngram_size_m': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
    'speculative.ngram_min_hits': 'per-request speculative settings are compiled out of llama.cpp 9f0e509 as well (#if 0)',
}
# still to build (phase 2+): explicit 400 until then, never a silent drop
LLAMA_PENDING = {
}
# kept for the contract / older clients
LLAMA_ONLY = LLAMA_SAMPLER | LLAMA_GATEWAY | frozenset(LLAMA_NOOP) | frozenset(LLAMA_PENDING)
ADAPTER_FIELDS = frozenset({"timings_per_token", "return_progress"})
ALIASES = {"n_predict": "max_tokens", "n_cmpl": "n"}
REASONING_FORMATS = ("none", "auto", "deepseek", "deepseek-legacy")


def _native_sets(native):
    """Accepted vLLM request fields: the live engine's OpenAPI lists when the gateway has read them,
    otherwise the lists pinned above (generated from the vLLM 0.32 schemas)."""
    if native and native.get("chat") and native.get("completion"):
        return frozenset(native["chat"]), frozenset(native["completion"]), native.get("source", "live")
    return VLLM_CHAT_NATIVE, VLLM_COMPLETION_NATIVE, "pinned fallback (vLLM 0.32 schemas)"


def engine_contract(engine, native=None):
    if engine not in ("llama.cpp", "vllm"):
        return {"version": 2, "engine": engine, "status": "unprobed"}
    chat_fields, completion_fields, source = _native_sets(native)
    if engine == "vllm":
        foreign = {
            "llama.cpp_sampler_chain": sorted(LLAMA_SAMPLER) + ["(+ temperature, top_k, top_p, min_p, frequency/presence_penalty with llama.cpp meaning in a dialect request)"],
            "llama.cpp_gateway": sorted(LLAMA_GATEWAY - frozenset(LLAMA_PENDING)),
            "llama.cpp_noop": dict(LLAMA_NOOP),
            "llama.cpp_pending": dict(LLAMA_PENDING),
        }
    else:
        foreign = "vLLM-only controls: implemented by the CT108 router and server patch 0053 (see PARITY-MATRIX.md)"
    return {
        "version": 2, "engine": engine,
        "request_controls": {
            "native": sorted(chat_fields) if engine == "vllm" else "llama.cpp native API; model /props declares active defaults",
            "native_completion": sorted(completion_fields) if engine == "vllm" else "llama.cpp native /completion API",
            "native_source": source if engine == "vllm" else "llama.cpp",
            "foreign": foreign,
            "mapped": dict(ALIASES) if engine == "vllm" else {},
            "adapter": sorted(ADAPTER_FIELDS),
            "unsupported": sorted(LLAMA_PENDING) if engine == "vllm" else [],
            "unsupported_policy": "explicit error; no silent removal; no-ops reported in the response `compat` block",
        },
        "reasoning": "model chat template; router reasoning_effort mapping is published separately",
        "prefix_cache": "engine managed; cache_prompt=false -> per-request cache_salt" if engine == "vllm" else "native cache_prompt / slot controls",
        "context_scope": "per_request" if engine == "vllm" else "runtime per-slot limit; not the summed allocation",
        "server_controls": "engine launch configuration; never sent as request sampling options",
        "telemetry": {"tokens": "engine usage, not SSE frame count", "live_prefill": "estimated" if engine == "vllm" else "native progress", "decode_clock": "engine when available; fallback explicitly estimated"},
    }


def translate_vllm(body, model_id=None, chat=True, native=None, gen_defaults=None):
    """Validate first, then mutate: rejected requests never get partially translated.
    Returns (timings, progress, post) where post carries the gateway-side behaviours (reasoning_format, response_fields,
    return_tokens, t_max_predict_ms, n_indent, sse_ping_interval, n_probs shaping) and the `compat` report."""
    if not isinstance(body, dict):
        raise CompatibilityError("Request body must be a JSON object")
    chat_fields, completion_fields, _source = _native_sets(native)
    accepted = (chat_fields if chat else completion_fields) | LLAMA_ONLY | frozenset(ALIASES)
    unknown = sorted(set(body) - accepted)
    if unknown:
        raise CompatibilityError("Unknown request controls: " + ", ".join(unknown))
    pending = sorted(k for k in body if k in LLAMA_PENDING and not (k == "post_sampling_probs" and not body[k]))
    if pending:
        raise CompatibilityError("Not implemented on vLLM yet: " + "; ".join(k + " (" + LLAMA_PENDING[k] + ")" for k in pending))
    if model_id and body.get("model") not in (None, model_id):
        raise CompatibilityError("Requested model is not the model served by this vLLM gateway")
    if "n_predict" in body:
        v = body["n_predict"]
        if type(v) is not int or (v <= 0 and v != -1):
            raise CompatibilityError("n_predict must be a positive integer or -1")
        for k in ("max_tokens", "max_completion_tokens"):
            if body.get(k) is not None and body[k] != v:
                raise CompatibilityError("Conflicting controls: n_predict and " + k)
    if "n_cmpl" in body and "n" in body and body["n"] != body["n_cmpl"]:
        raise CompatibilityError("Conflicting controls: n_cmpl and n")
    rf = body.get("reasoning_format")
    if rf is not None and rf not in REASONING_FORMATS:
        raise CompatibilityError("reasoning_format must be one of " + ", ".join(REASONING_FORMATS))
    lora = body.get("lora")
    if lora and any((isinstance(a, dict) and a.get("scale", 1.0) != 0.0) for a in lora):
        raise CompatibilityError("no LoRA adapters are loaded on this vLLM instance (llama.cpp answers the same for an unknown adapter id)")
    if body.get("grammar") is not None and body.get("json_schema") is not None:
        raise CompatibilityError("Conflicting controls: grammar and json_schema")
    dialect = any(k in body for k in LLAMA_SAMPLER)
    # llama.cpp-only behaviours that live in the processor select the llama.cpp dialect too
    lazy = bool(body.get("grammar_lazy"))
    if lazy and body.get("grammar") is None and body.get("json_schema") is None:
        raise CompatibilityError("grammar_lazy needs grammar or json_schema")
    lazy_grammar = None
    if lazy:
        lazy_grammar = body.get("grammar")
        if lazy_grammar is None:
            try:
                import json_schema_to_grammar as _J
                conv = _J.SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
                sch = conv.resolve_refs(body["json_schema"], "")
                conv.visit(sch, "")
                lazy_grammar = conv.format_grammar()
            except Exception as exc:
                raise CompatibilityError("json_schema could not be converted to a grammar: %s" % exc)
        from lab_llama_samplers import validate as _v
        try:
            _v({"grammar": lazy_grammar, "grammar_lazy": True, "grammar_triggers": body.get("grammar_triggers") or []})
        except ValueError as exc:
            raise CompatibilityError(str(exc))
    proc = bool(body.get("reasoning_budget_message")) or bool(body.get("reasoning_control")) or lazy or         (bool(body.get("post_sampling_probs")) and (body.get("n_probs") or 0) > 0)
    if proc and body.get("use_beam_search"):
        raise CompatibilityError("use_beam_search cannot be combined with reasoning_budget_message / reasoning_control / post_sampling_probs")
    if body.get("reasoning_budget_message") is not None and not isinstance(body.get("reasoning_budget_message"), str):
        raise CompatibilityError("reasoning_budget_message must be a string")
    dialect = dialect or proc
    if dialect and body.get("use_beam_search"):
        raise CompatibilityError("use_beam_search selects vLLM's beam search; it cannot be combined with llama.cpp sampler controls")
    if dialect:
        from lab_llama_samplers import validate
        try:
            validate({k: body[k] for k in LLAMA_SAMPLER | frozenset(SHARED_SAMPLER) | {"seed"} if k in body})
        except ValueError as exc:
            raise CompatibilityError(str(exc))

    # ---- mutate ------------------------------------------------------------------------------------------------
    notes = {k: LLAMA_NOOP[k] for k in body if k in LLAMA_NOOP}
    post = {"compat": {}}
    timings, progress = bool(body.pop("timings_per_token", False)), bool(body.pop("return_progress", False))
    for k in list(notes):
        body.pop(k, None)
    body.pop("lora", None)
    want_psp = bool(body.pop("post_sampling_probs", None))
    lazy_triggers = body.pop("grammar_triggers", None)
    body.pop("grammar_lazy", None)
    if lazy:                                      # the processor owns it; not vLLM's (non-lazy) structured outputs
        body.pop("grammar", None)
        body.pop("json_schema", None)
    rb_message = body.pop("reasoning_budget_message", None) or ""
    rb_control = bool(body.pop("reasoning_control", False))
    if "n_predict" in body:
        v = body.pop("n_predict")
        if v != -1 and body.get("max_completion_tokens") is None and body.get("max_tokens") is None:
            body["max_tokens"] = v
    if "n_cmpl" in body:
        body.setdefault("n", body.pop("n_cmpl"))
    if body.pop("cache_prompt", True) is False and "cache_salt" not in body:
        body["cache_salt"] = "nocache-" + uuid.uuid4().hex          # llama.cpp cache_prompt=false: no prefix reuse
    n_probs = body.pop("n_probs", 0) or 0
    if n_probs > 0:
        if chat:
            body.setdefault("logprobs", True)
            body.setdefault("top_logprobs", n_probs)
        else:
            body.setdefault("logprobs", n_probs)
        post["n_probs"] = n_probs
    if body.get("grammar") is not None or body.get("json_schema") is not None:
        so = dict(body.get("structured_outputs") or {})
        if body.get("grammar") is not None:
            so["grammar"] = body.pop("grammar")
        else:
            body.pop("grammar", None)
        if body.get("json_schema") is not None:
            so["json"] = body.pop("json_schema")
        else:
            body.pop("json_schema", None)
        body["structured_outputs"] = so
    rb_budget = -1
    for k in ("reasoning_budget_tokens", "thinking_budget_tokens"):
        if k in body:
            v = body.pop(k)
            if v is not None and v >= 0:
                rb_budget = v
    if rb_budget >= 0 and not (rb_message or rb_control):
        body.setdefault("thinking_token_budget", rb_budget)       # vLLM's native budget is the same machine
    if rf is not None or "reasoning_in_content" in body:
        post["reasoning_format"] = "none" if body.pop("reasoning_in_content", False) else (body.get("reasoning_format") or "auto")
        body.pop("reasoning_format", None)
    for k in ("return_tokens", "response_fields", "t_max_predict_ms", "n_indent", "sse_ping_interval", "verbose"):
        if k in body:
            post[k] = body.pop(k)
    if post.get("return_tokens"):
        body["return_token_ids"] = True
    if dialect:
        cfg = {k: body.pop(k) for k in list(body) if k in LLAMA_SAMPLER}
        for k in SHARED_SAMPLER:
            if k in body:
                cfg[k] = body.pop(k)
        gd = gen_defaults or {}
        for k in ("temperature", "top_k", "top_p", "min_p"):    # the model's own defaults, as vLLM would use them
            if k not in cfg and gd.get(k) is not None:
                cfg[k] = gd[k]
        if body.get("seed") is not None:
            cfg["seed"] = body["seed"]
        if n_probs:
            cfg["n_probs"] = n_probs
        # every dialect request gets a lab request id: response id, control/side-channel key, and the seed source of
        # an unseeded request (identical on every tensor-parallel rank)
        rid = body.get("request_id") if isinstance(body.get("request_id"), str) and body["request_id"].replace("-", "").replace("_", "").isalnum() else uuid.uuid4().hex[:24]
        body["request_id"] = rid
        cfg["lab_rid"] = rid
        post["lab_rid"] = rid
        if rb_message or rb_control:
            cfg["reasoning_budget_tokens"] = rb_budget
            if rb_message:
                cfg["reasoning_budget_message"] = rb_message
            if rb_control:
                cfg["reasoning_control"] = True
                post["reasoning_control"] = True
        if lazy:
            cfg.update(grammar=lazy_grammar, grammar_lazy=True, grammar_triggers=lazy_triggers)
        if want_psp and n_probs:
            cfg["post_sampling_probs"] = True
            post["psp"] = n_probs
            body["return_tokens_as_token_ids"] = True              # ids to look the post-chain probabilities up
        xa = dict(body.get("vllm_xargs") or {})
        import json as _json
        xa["llama"] = _json.dumps(cfg, separators=(",", ":"))   # vllm_xargs values must be scalars/lists
        body["vllm_xargs"] = xa
        # the chain is complete in the processor: vLLM's own stages must be neutral (final draw = softmax sample)
        body.update(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0,
                    frequency_penalty=0.0, presence_penalty=0.0)
        post["compat"]["sampler"] = "llama.cpp chain (lab_llama_samplers)"
    if notes:
        post["compat"]["noop"] = notes
    if model_id:
        body["model"] = model_id
    return timings, progress, post
