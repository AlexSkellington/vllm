"""lab_llama_samplers.py (2026-11-07, Claude; 2026-11-17 sync-free early exit) - llama.cpp's sampler chain as a vLLM Model Runner V2 logits processor.

Engine parity (user: "support natively both engines completely", "completely parity"): a request that uses llama.cpp's
own sampling controls gets llama.cpp's sampling on vLLM. The AS1 gateway resolves such a request into
`vllm_xargs = {"llama": {...}}` (every key llama.cpp's server accepts, with llama.cpp's defaults for unset ones) and
neutralises vLLM's own temperature/top-k/top-p/min-p/penalties, so this processor is the whole chain and vLLM's final
draw is a plain softmax sample (llama.cpp's `dist`).

Ported from llama.cpp 9f0e509 (src/llama-sampler.cpp, common/sampling.cpp; MIT):
  chain       logit_bias (vLLM native, same) -> [penalties, dry, top_n_sigma, top_k, typ_p, top_p, min_p, xtc, temperature]
              in `samplers` order (default above) -> dist | greedy | adaptive_p;  mirostat 1/2 replace the chain with
              temp -> mirostat (exactly as common_sampler_init does).
  candidates  llama.cpp shrinks the candidate array (top_k/top_p/min_p/typical/xtc/mirostat); -INF from logit_bias,
              grammar or top_n_sigma stays IN the array. Both are tracked here (`alive` vs finite logits), because
              dynatemp's max entropy and top_n_sigma's statistics depend on the difference.
  history     penalties and DRY see the prompt too (server init_sampler accepts every prompt token) plus the output;
              under speculative decoding each draft row sees the committed tokens plus the draft tokens before it.
  feedback    mirostat mu and adaptive-p's EMA advance on each committed token using that token's probability in the
              distribution its row produced (stored per position); draft rows advance a temporary copy.
Differences that cannot be removed: llama.cpp's mt19937 stream (XTC coin, final draw) is not reproduced bit-for-bit;
distributions are identical. Typical-p skips p=0 terms where llama.cpp would compute 0*log(0)=NaN.
"""
from __future__ import annotations

import math
import random

try:  # the engine (vLLM main, Model Runner V2) has the interface; the gateway (older venv) only needs validate()
    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor
except Exception:  # pragma: no cover
    class LogitsProcessor:  # type: ignore[no-redef]
        pass

KEY = "llama"
DEFAULT_ORDER = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]
DEFAULTS = {  # common/common.h common_params_sampling (9f0e509)
    "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "typical_p": 1.0, "top_n_sigma": -1.0,
    "xtc_probability": 0.0, "xtc_threshold": 0.1, "min_keep": 0, "repeat_last_n": 64, "repeat_penalty": 1.0,
    "frequency_penalty": 0.0, "presence_penalty": 0.0, "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2,
    "dry_penalty_last_n": 64, "dry_sequence_breakers": ["\n", ":", "\"", "*"], "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0, "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "adaptive_target": -1.0,
    "adaptive_decay": 0.9, "samplers": DEFAULT_ORDER, "seed": -1, "n_probs": 0,
    # 2026-11-08 (parity 3): llama.cpp reasoning budget with message / runtime control, post-sampling probabilities
    "reasoning_budget_tokens": -1, "reasoning_budget_message": "", "reasoning_control": False,
    "reasoning_start": "<think>", "reasoning_end": "</think>", "post_sampling_probs": False, "lab_rid": "",
    # llama.cpp lazy grammar (grammar + grammar_lazy + grammar_triggers [{type 0 token|1 word|2 pattern|3 pattern_full,
    # value, token}]); a non-lazy grammar goes to vLLM's own structured outputs instead
    "grammar": "", "grammar_lazy": False, "grammar_triggers": [],
}
CONTROL_DIR = "/dev/shm/lab-control"   # gateway drops <rid> here for POST /v1/chat/completions/control reasoning_end
PSP_DIR = "/dev/shm/lab-psp"           # TP rank 0 appends "<output index>\t<id>:<p>,..." per row (last write wins)
RB_IDLE, RB_COUNTING, RB_FORCING, RB_DONE, RB_WAIT = 0, 1, 2, 3, 4  # RB_WAIT = WAITING_UTF8
_CANON = ["dry", "top_k", "top_p", "top_n_sigma", "typ_p", "min_p", "temperature", "xtc", "infill", "penalties", "adaptive_p"]
NAME_MAP = {}
for _c in _CANON:
    NAME_MAP[_c] = _c
    if "_" in _c:
        NAME_MAP[_c.replace("_", "-")] = _c
        NAME_MAP[_c.replace("_", "")] = _c
NAME_MAP.update({"nucleus": "top_p", "temp": "temperature", "typ": "typ_p"})
CHAR_MAP = {"d": "dry", "k": "top_k", "y": "typ_p", "p": "top_p", "s": "top_n_sigma", "m": "min_p", "t": "temperature",
            "x": "xtc", "i": "infill", "e": "penalties", "a": "adaptive_p"}
NEG_INF = float("-inf")
FLOAT_MAX_LOG = 88.7228391


def parse_samplers(v):
    """common_sampler_types_from_names / _from_chars: unknown names are skipped (llama.cpp only warns)."""
    if isinstance(v, str):
        return [CHAR_MAP[c] for c in v if c in CHAR_MAP]
    return [NAME_MAP[n.lower()] for n in v if isinstance(n, str) and n.lower() in NAME_MAP]


def resolve(cfg):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in cfg.items() if v is not None})
    p["samplers"] = parse_samplers(cfg["samplers"]) if cfg.get("samplers") is not None else list(DEFAULT_ORDER)
    if p["dry_base"] < 1.0:  # server-schema.cpp: values < 1.0 are replaced with the default
        p["dry_base"] = DEFAULTS["dry_base"]
    if "infill" in p["samplers"]:
        raise ValueError("llama.cpp 'infill' sampler needs llama.cpp's FIM vocabulary handling; use /infill on a llama.cpp backend")
    return p


def validate(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("vllm_xargs.llama must be an object")
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown llama.cpp sampler keys: " + ", ".join(sorted(unknown)))
    p = resolve(cfg)
    if p["mirostat"] not in (0, 1, 2):
        raise ValueError("mirostat must be 0, 1 or 2")
    if p["repeat_last_n"] < -1 or p["dry_penalty_last_n"] < -1 or p["dry_allowed_length"] < 0:
        raise ValueError("repeat_last_n / dry_penalty_last_n must be >= -1, dry_allowed_length >= 0")
    if not isinstance(p["dry_sequence_breakers"], list) or not p["dry_sequence_breakers"]:
        raise ValueError("dry_sequence_breakers must be a non-empty array of strings")
    if not isinstance(p["reasoning_budget_tokens"], int) or p["reasoning_budget_tokens"] < -1:
        raise ValueError("reasoning_budget_tokens must be an integer >= -1")
    if p["lab_rid"] and not all(c.isalnum() or c in "-_" for c in p["lab_rid"]):
        raise ValueError("lab_rid must be alphanumeric")
    if p["grammar_lazy"]:
        if not isinstance(p["grammar"], str) or "::=" not in p["grammar"]:
            raise ValueError("grammar_lazy needs a GBNF grammar")
        trig = p["grammar_triggers"]
        if not isinstance(trig, list) or not trig:
            raise ValueError("Error: no triggers set for lazy grammar!")
        for t in trig:
            if not isinstance(t, dict) or t.get("type") not in (0, 1, 2, 3) or not isinstance(t.get("value"), str) \
                    or (t["type"] == 0 and not isinstance(t.get("token"), int)):
                raise ValueError("grammar_triggers: each trigger is {type: 0..3, value: string, token: int for type 0}")
            if t["type"] in (2, 3):
                import re
                try:
                    re.compile(t["value"].encode())
                except re.error as exc:
                    raise ValueError("grammar_triggers: invalid pattern: %s" % exc)
    return p


def _cfg_of(sampling_params):
    """vllm_xargs.llama arrives as a JSON string (vLLM accepts only scalar/list xargs values)."""
    cfg = (sampling_params.extra_args or {}).get(KEY)
    if isinstance(cfg, str):
        import json
        cfg = json.loads(cfg)
    return cfg


class LlamaSamplerChain(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        cfg = _cfg_of(sampling_params)
        if cfg is not None:
            validate(cfg)

    def __init__(self, vllm_config, req_states):
        import torch
        self.torch = torch
        self.rs = req_states
        self.vocab = req_states.vocab_size
        self.max_ctx = vllm_config.model_config.max_model_len
        self.tok_name = vllm_config.model_config.tokenizer
        self._tokenizer = None
        self._vocab_bytes = None
        self._breaker_cache = {}
        self.cfg = {}        # slot -> resolved params
        self.state = {}      # slot -> dict(mu, ws, tw, rng, committed_pos, rows={pos: probs tensor})
        self._tp = None      # (rank_in_group, world_size, cpu_group) of the tensor-parallel group, resolved lazily
        self._tag_ids = {}   # tag string -> token ids
        self._gcomp = None   # xgrammar GrammarCompiler (lazy grammars), built on first use
        self._gcache = {}    # grammar text -> CompiledGrammar

    # ---- vocab helpers (DRY sequence breakers) -------------------------------------------------------------------
    def _tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tok_name, trust_remote_code=True)
        return self._tokenizer

    def _vocab_words(self):
        """Raw bytes of every token, like vocab->detokenize({id}, true) (byte-level BPE inverse map)."""
        if self._vocab_bytes is None:
            tok = self._tok()
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs, n = bs[:], 0
            for b in range(256):
                if b not in bs:
                    bs.append(b); cs.append(256 + n); n += 1
            dec = {chr(c): b for b, c in zip(bs, cs)}
            words = []
            for t in tok.convert_ids_to_tokens(list(range(len(tok)))):
                t = t or ""
                if all(ch in dec for ch in t):
                    words.append(bytes(dec[ch] for ch in t))
                else:  # special / added tokens are stored as text
                    words.append(t.encode("utf-8"))
            self._vocab_bytes = words
        return self._vocab_bytes

    def _breakers(self, strings):
        """get_overlapping_token_sequences for each breaker (MAX_CHAR_LEN 40, MAX_SEQ_LEN 20), cached per string set."""
        key = tuple(strings)
        if key in self._breaker_cache:
            return self._breaker_cache[key]
        words, tok = self._vocab_words(), self._tok()
        seqs, seen = {}, set()
        for s in strings:
            if not s:
                continue
            sb = s.encode("utf-8")[:40]
            if sb in seen:
                continue
            seen.add(sb)
            tail_cache = {}
            for tid, w in enumerate(words):
                if sb in w:
                    lst = seqs.setdefault(tid, [])
                    if [] not in lst:
                        lst.append([])
                    continue
                pos = w.find(sb[:1])
                while pos != -1:
                    i, ok = 1, True
                    while i < len(sb) and i + pos < len(w):
                        if w[pos + i] != sb[i]:
                            ok = False
                            break
                        i += 1
                    if ok:
                        if i not in tail_cache:
                            tail_cache[i] = tok.encode(sb[i:].decode("utf-8", "ignore"), add_special_tokens=False)[:20]
                        lst = seqs.setdefault(tid, [])
                        if tail_cache[i] not in lst:
                            lst.append(tail_cache[i])
                    pos = w.find(sb[:1], pos + 1)
        self._breaker_cache[key] = seqs
        return seqs

    # ---- V2 interface ----------------------------------------------------------------------------------------------
    def add_request(self, req_idx, sampling_params):
        cfg = _cfg_of(sampling_params)
        self.state.pop(req_idx, None)
        if cfg is None:
            self.cfg.pop(req_idx, None)
            return False
        p = resolve(cfg)
        if p["dry_multiplier"] != 0.0 and p["dry_penalty_last_n"] != 0:
            p["_breakers"] = self._breakers(p["dry_sequence_breakers"])
        seed = p["seed"]
        unseeded = seed is None or seed < 0 or seed == 0xFFFFFFFF
        self.cfg[req_idx] = p
        self.state[req_idx] = {
            "mu": 2.0 * p["mirostat_tau"],
            "ws": p["adaptive_target"] / (1.0 - p["adaptive_decay"]) if p["adaptive_decay"] < 1 else 0.0,
            "tw": 1.0 / (1.0 - p["adaptive_decay"]) if p["adaptive_decay"] < 1 else 0.0,
            # every TP rank runs this processor and must draw the same XTC coins: an unseeded request is seeded from its
            # lab request id (same on every rank, different per request), never from os.urandom
            "rng": random.Random(("lab-rid:" + p["lab_rid"]) if unseeded and p["lab_rid"] else (None if unseeded else seed)),
            "committed": None, "rows": {}, "rb": None,
        }
        self.state[req_idx]["g"] = None
        if p["grammar_lazy"]:
            import re
            tokens, pats = set(), []
            for t in p["grammar_triggers"]:
                if t["type"] == 0:
                    tokens.add(t["token"])
                elif t["type"] == 1:
                    ids = self._ids(t["value"])
                    if len(ids) == 1:               # common_sampling_add_grammar_triggers: single-token word -> token
                        tokens.add(ids[0])
                    else:
                        pats.append(re.escape(t["value"].encode()))
                elif t["type"] == 2:
                    pats.append(t["value"].encode())
                else:
                    v = t["value"].encode()
                    pats.append((b"" if v.startswith(b"^") else b"^") + v + (b"" if v.endswith(b"$") else b"$") if v else b"^$")
            self.state[req_idx]["g"] = {"compiled": self._compile(p["grammar"]), "tokens": tokens,
                                        "pats": [(re.compile(x), x.startswith(b"^") and x.endswith(b"$")) for x in pats],
                                        "awaiting": True, "buf": b"", "m": None, "dead": False, "committed": None}
        if p["reasoning_budget_tokens"] >= 0 or p["reasoning_control"] or p["grammar_lazy"]:
            budget = p["reasoning_budget_tokens"] if p["reasoning_budget_tokens"] >= 0 else (1 << 31) - 1  # INT_MAX
            end = self._ids(p["reasoning_end"])
            self.state[req_idx]["rb"] = {
                "init": False, "committed": None, "budget": budget, "state": RB_IDLE, "remaining": budget,
                "force_pos": 0, "tail": [], "start": self._ids(p["reasoning_start"]), "end": end,
                "forced": (self._ids(p["reasoning_budget_message"]) if p["reasoning_budget_message"] else []) + end,
            }
        return True

    # ---- reasoning budget (common/reasoning-budget.cpp, 9f0e509) ------------------------------------------------------
    def _ids(self, text):
        if text not in self._tag_ids:
            self._tag_ids[text] = list(self._tok().encode(text, add_special_tokens=False)) if text else []
        return self._tag_ids[text]

    def _utf8_complete(self, tok):
        words = self._vocab_words()
        w = words[tok] if 0 <= tok < len(words) else b""
        for back in range(1, min(4, len(w)) + 1):  # common_utf8_is_complete: the last lead byte has its continuation
            c = w[-back]
            if c & 0xC0 == 0x80:
                continue
            need = 1 if c < 0x80 else 2 if c >> 5 == 0x6 else 3 if c >> 4 == 0xE else 4 if c >> 3 == 0x1E else 1
            return back >= need
        return True

    def _rb_accept(self, rb, tok):
        """One accepted token through the budget state machine (IDLE/COUNTING/WAITING_UTF8/FORCING/DONE)."""
        rb["tail"] = (rb["tail"] + [tok])[-32:]
        ends = lambda seq: bool(seq) and rb["tail"][-len(seq):] == seq
        s = rb["state"]
        if s in (RB_IDLE, RB_DONE):
            if ends(rb["start"]):
                rb["state"], rb["remaining"], rb["tail"] = RB_COUNTING, rb["budget"], []
                if rb["remaining"] <= 0:
                    rb["state"], rb["force_pos"] = RB_FORCING, 0
        elif s in (RB_COUNTING, RB_WAIT):
            if ends(rb["end"]):
                rb["state"], rb["tail"] = RB_DONE, []
                return
            complete = self._utf8_complete(tok)
            if s == RB_WAIT:
                if complete:
                    rb["state"], rb["force_pos"], rb["tail"] = RB_FORCING, 0, []
            else:
                rb["remaining"] -= 1
                if rb["remaining"] <= 0:
                    rb["state"], rb["tail"] = (RB_FORCING if complete else RB_WAIT), []
                    rb["force_pos"] = 0
        elif s == RB_FORCING:
            rb["force_pos"] += 1
            if ends(rb["end"]) or rb["force_pos"] >= len(rb["forced"]):
                rb["state"], rb["tail"] = RB_DONE, []

    def _rb_commit(self, slot, rb, pos0, g=None):
        """Advance the committed budget state to the last committed token (prompt tail = llama.cpp's prefill accept)."""
        if not rb["init"]:
            plen = int(self.rs.prompt_len.np[slot])
            for t in self.rs.all_token_ids.gpu[slot, max(0, plen - 16):plen].tolist():
                self._rb_accept(rb, t)
            rb["init"], rb["committed"] = True, plen - 1
        if pos0 > rb["committed"]:
            for t in self.rs.all_token_ids.gpu[slot, rb["committed"] + 1:pos0 + 1].tolist():
                if g is not None:
                    self._g_feed(g, rb, t)
                else:
                    self._rb_accept(rb, t)
            rb["committed"] = pos0

    # ---- lazy grammar (src/llama-grammar.cpp lazy triggers + common_sampler grammar-last rejection sampling) -----------
    def _compile(self, text):
        if text not in self._gcache:
            import xgrammar as xgr
            if self._gcomp is None:
                info = xgr.TokenizerInfo.from_huggingface(self._tok(), vocab_size=self.vocab)
                self._gcomp = xgr.GrammarCompiler(info)
            self._gcache[text] = self._gcomp.compile_grammar(xgr.Grammar.from_ebnf(text))
        return self._gcache[text]

    @staticmethod
    def _g_copy(g):
        return dict(g, m=g["m"].fork() if g["m"] is not None else None)

    def _g_accept(self, g, tok):
        """llama_grammar_accept_impl for a lazy grammar: buffer text until a trigger, then constrain from its start."""
        if g["dead"]:
            return
        if g["awaiting"]:
            import xgrammar as xgr
            if tok in g["tokens"]:
                g["awaiting"], g["buf"] = False, b""
                g["m"] = xgr.GrammarMatcher(g["compiled"])
                g["dead"] = not g["m"].accept_token(tok)
                return
            words = self._vocab_words()
            g["buf"] += words[tok] if 0 <= tok < len(words) else b""
            for rx, full in g["pats"]:
                mt = rx.fullmatch(g["buf"]) if full else rx.search(g["buf"])
                if mt is None:
                    continue
                start = next((mt.start(i) for i in range(1, (mt.lastindex or 0) + 1) if mt.group(i)), mt.start())
                g["awaiting"] = False
                g["m"] = xgr.GrammarMatcher(g["compiled"])
                g["dead"] = not g["m"].accept_string(g["buf"][start:])
                g["buf"] = b""
                return
            return
        if g["m"].is_terminated():           # complete (stop token accepted): nothing left to constrain
            return
        try:
            g["dead"] = not g["m"].accept_token(tok)
        except Exception:
            g["dead"] = True

    def _g_apply_ok(self, rb):
        """grammar_should_apply for a lazy grammar: only outside the reasoning block."""
        return rb is None or rb["state"] in (RB_IDLE, RB_DONE)

    def _g_feed(self, g, rb, tok):
        """common_sampler_accept order: decide on the budget state BEFORE it sees the token; replay the end sequence."""
        apply = self._g_apply_ok(rb)
        was_done = rb is not None and rb["state"] == RB_DONE
        if rb is not None:
            self._rb_accept(rb, tok)
        if apply:
            self._g_accept(g, tok)
        elif rb is not None and rb["state"] == RB_DONE and not was_done:
            for t in rb["end"]:
                self._g_accept(g, t)

    def _g_mask(self, g, x):
        """0 for grammar-valid tokens, -inf otherwise (None when the grammar is not constraining)."""
        if g["awaiting"] or g["dead"] or g["m"] is None or g["m"].is_terminated():
            return None
        import xgrammar as xgr
        bm = xgr.allocate_token_bitmask(1, self.vocab)
        try:
            g["m"].fill_next_token_bitmask(bm, 0)
        except Exception:
            g["dead"] = True
            return None
        mask = self.torch.zeros((1, x.numel()), device=x.device, dtype=x.dtype)
        xgr.apply_token_bitmask_inplace(mask, bm.to(x.device), vocab_size=self.vocab)
        return mask[0]

    # ---- tensor-parallel agreement: every rank samples, so runtime signals must be decided once and broadcast ---------
    def _tp_info(self):
        if self._tp is None:
            try:
                from vllm.distributed import get_tp_group
                g = get_tp_group()
                self._tp = (g.rank_in_group, g.world_size, g.cpu_group, g.ranks[0])
            except Exception:
                self._tp = (0, 1, None, 0)
        return self._tp

    def _control_flags(self, slots):
        import os
        rank, world, grp, src = self._tp_info()
        flags = [0] * len(slots)
        if rank == 0:
            for i, s in enumerate(slots):
                f = os.path.join(CONTROL_DIR, self.cfg[s]["lab_rid"])
                if self.cfg[s]["lab_rid"] and os.path.exists(f):
                    flags[i] = 1
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        if world > 1:
            t = self.torch.tensor(flags, dtype=self.torch.int32)
            self.torch.distributed.broadcast(t, src=src, group=grp)
            flags = t.tolist()
        return flags

    def apply(self, logits, ctx):
        # 2026-11-17: sync-free early exit. vLLM calls every processor whenever ANY request in the batch needs logits
        # processing (plain temperature/top-p/top-k sampling does); the .tolist() below is a blocking GPU->CPU copy, so
        # check the CPU-side mapping first and only touch GPU tensors when a llama.cpp-chain request is in the batch.
        cfg = self.cfg
        if not cfg:
            return logits
        for s in ctx.idx_mapping_np:
            if int(s) in cfg:
                break
        else:
            return logits
        torch = self.torch
        slots = ctx.expanded_idx_mapping.tolist()
        mine = [i for i, s in enumerate(slots) if s in self.cfg]
        if not mine:
            return logits
        local = ctx.expanded_local_pos.tolist()
        pos = ctx.pos.tolist()
        ids = ctx.input_ids.tolist()
        by_slot = {}
        for r in mine:
            by_slot.setdefault(slots[r], []).append(r)
        # committed tokens end at total_len (vLLM's own bad_words kernel uses the same convention); row 0's input token is
        # the last committed one, draft tokens are the input ids of rows 1..
        pos0s = {slot: int(self.rs.total_len.gpu[slot]) - 1 for slot in by_slot}
        for slot in by_slot:
            rb = self.state[slot]["rb"]
            if rb is not None:
                self._rb_commit(slot, rb, pos0s[slot], self.state[slot]["g"])
        ctl = [slot for slot in by_slot if self.cfg[slot]["reasoning_control"]]
        if ctl:  # same slots on every rank; rank 0 reads the gateway's signal, all ranks act on the same step
            for slot, f in zip(ctl, self._control_flags(ctl)):
                rb = self.state[slot]["rb"]
                if f and rb["state"] == RB_COUNTING:  # common_reasoning_budget_force: only while counting
                    rb["state"], rb["force_pos"], rb["tail"] = RB_FORCING, 0, []
        rank = None
        for slot, rows in by_slot.items():
            try:
                rank = self._apply_slot(logits, slot, rows, local, pos, ids, pos0s, rank)
            except Exception as exc:  # never take the engine down: the request falls back to plain logits (same on every rank)
                import logging
                logging.getLogger("lab_llama_samplers").exception("slot %s: llama.cpp chain disabled after error: %s", slot, exc)
                self.cfg.pop(slot, None)
        return logits

    def _apply_slot(self, logits, slot, rows, local, pos, ids, pos0s, rank):
        torch = self.torch
        if True:
            rows.sort(key=lambda r: local[r])
            p, st = self.cfg[slot], self.state[slot]
            pos0 = pos0s[slot]
            hist_n = self._hist_needed(p)
            start = max(0, pos0 + 1 - hist_n) if hist_n >= 0 else 0
            committed = self.rs.all_token_ids.gpu[slot, start:pos0 + 1].tolist() if hist_n != 0 else []
            self._commit_feedback(p, st, slot, pos0)
            mu, ws, tw = st["mu"], st["ws"], st["tw"]
            rb = dict(st["rb"], tail=list(st["rb"]["tail"])) if st["rb"] is not None else None
            gc = self._g_copy(st["g"]) if st["g"] is not None else None
            psp = []
            new_rows = {}
            for j, r in enumerate(rows):
                if j > 0 and gc is not None:
                    self._g_feed(gc, rb, ids[rows[j]])  # the draft token this row is conditioned on
                elif j > 0 and rb is not None:
                    self._rb_accept(rb, ids[rows[j]])
                if rb is not None and rb["state"] == RB_FORCING and rb["force_pos"] < len(rb["forced"]):
                    x = torch.full_like(logits[r], NEG_INF)  # llama.cpp: every candidate but the forced token is -inf
                    x[rb["forced"][rb["force_pos"]]] = 0.0
                    probs_for_fb = None
                else:
                    hist = committed + [ids[rr] for rr in rows[1:j + 1]]
                    x, probs_for_fb = self._chain(logits[r], p, st, hist, mu, ws, tw)
                    mask = self._g_mask(gc, x) if gc is not None and self._g_apply_ok(rb) else None
                    if mask is not None:
                        # grammar_first=false: sample from the chain, keep a grammar-valid token, otherwise re-sample from
                        # chain(grammar(logits)). As one distribution: P = q1*valid + (1 - V) * q2
                        q1 = torch.softmax(x.float(), dim=-1)
                        valid = mask == 0
                        V = float(q1[valid].sum())
                        if V >= 1.0 - 1e-6:
                            x = x + mask
                        else:
                            x2, _ = self._chain(logits[r] + mask, p, st, hist, mu, ws, tw)
                            P = q1 * valid + (1.0 - V) * torch.softmax(x2.float(), dim=-1)
                            x = torch.log(P).to(logits.dtype)
                logits[r] = x
                new_rows[pos[r]] = probs_for_fb
                if p["post_sampling_probs"] and p["lab_rid"]:
                    if rank is None:
                        rank = self._tp_info()[0]
                    if rank == 0:
                        pr = torch.softmax(x.float(), dim=-1)
                        kk = max(1, min(256, int((pr > 0).sum())))
                        v, i = torch.topk(pr, kk)
                        out_idx = pos[r] + 1 - int(self.rs.prompt_len.np[slot])
                        psp.append("%d\t%s" % (out_idx, ",".join("%d:%.7g" % (a, b) for a, b in zip(i.tolist(), v.tolist()))))
                if j + 1 < len(rows) and probs_for_fb is not None:  # draft token of the next row, under this row
                    mu, ws, tw = self._advance(p, mu, ws, tw, probs_for_fb, ids[rows[j + 1]])
            st["rows"] = new_rows
            if psp:
                import os
                os.makedirs(PSP_DIR, exist_ok=True)
                with open(os.path.join(PSP_DIR, p["lab_rid"]), "a") as fh:
                    fh.write("\n".join(psp) + "\n")
        return rank

    # ---- feedback (mirostat mu / adaptive-p EMA) -------------------------------------------------------------------
    def _commit_feedback(self, p, st, slot, pos0):
        """Advance the committed state over tokens committed since the last step (each predicted by a stored row)."""
        if p["mirostat"] == 0 and not (p["adaptive_target"] >= 0 and "adaptive_p" in p["samplers"]):
            return
        last = st["committed"]
        st["committed"] = pos0
        if last is None or not st["rows"]:
            return
        toks = self.rs.all_token_ids.gpu[slot, last + 1:pos0 + 1].tolist()
        for k, t in enumerate(toks):
            probs = st["rows"].get(last + k)
            if probs is None:
                break
            st["mu"], st["ws"], st["tw"] = self._advance(p, st["mu"], st["ws"], st["tw"], probs, t)

    def _advance(self, p, mu, ws, tw, probs, token):
        if p["mirostat"]:
            pt = float(probs[token])
            surprise = -math.log2(pt) if pt > 0 else 1e9
            mu = mu - p["mirostat_eta"] * (surprise - p["mirostat_tau"])
        elif p["adaptive_target"] >= 0:
            ws = float(probs[token]) + p["adaptive_decay"] * ws
            tw = 1.0 + p["adaptive_decay"] * tw
        return mu, ws, tw

    def _hist_needed(self, p):
        n = 0
        if "penalties" in p["samplers"] and p["mirostat"] == 0:
            rl = self.max_ctx if p["repeat_last_n"] == -1 else p["repeat_last_n"]
            if not (rl == 0 or (p["repeat_penalty"] == 1.0 and p["frequency_penalty"] == 0.0 and p["presence_penalty"] == 0.0)):
                n = max(n, rl)
        if "dry" in p["samplers"] and p["mirostat"] == 0 and p["dry_multiplier"] != 0.0 and p["dry_penalty_last_n"] != 0:
            n = max(n, self.max_ctx if p["dry_penalty_last_n"] == -1 else p["dry_penalty_last_n"])
        return n

    # ---- the chain on one row ----------------------------------------------------------------------------------------
    def _chain(self, x, p, st, hist, mu, ws, tw):
        torch = self.torch
        x = x.clone()
        alive = torch.ones_like(x, dtype=torch.bool)
        fb = None
        if p["mirostat"]:
            x = self._temp(x, alive, p["temperature"])
            x, fb = self._mirostat(x, alive, p, mu)
            return x, fb
        for name in p["samplers"]:
            if name == "penalties":
                self._penalties(x, p, hist)
            elif name == "dry":
                self._dry(x, p, hist)
            elif name == "top_n_sigma":
                self._top_n_sigma(x, alive, p["top_n_sigma"])
            elif name == "top_k":
                self._top_k(x, alive, p["top_k"])
            elif name == "typ_p":
                self._typical(x, alive, p["typical_p"], p["min_keep"])
            elif name == "top_p":
                self._top_p(x, alive, p["top_p"], p["min_keep"])
            elif name == "min_p":
                self._min_p(x, alive, p["min_p"], p["min_keep"])
            elif name == "xtc":
                self._xtc(x, alive, p, st["rng"])
            elif name == "temperature":
                x = self._temp_ext(x, alive, p["temperature"], p["dynatemp_range"], p["dynatemp_exponent"])
        if "adaptive_p" in p["samplers"]:
            x, fb = self._adaptive_p(x, alive, p, ws, tw)
        return x, fb

    # helpers: candidates = alive entries; -inf logits may be alive (llama.cpp keeps them in the array)
    def _softmax(self, x, alive):
        torch = self.torch
        z = torch.where(alive, x, torch.full_like(x, NEG_INF))
        return torch.softmax(z, dim=-1)

    def _kill(self, x, alive, keep):
        alive &= keep
        x[~alive] = NEG_INF

    def _temp(self, x, alive, t):
        torch = self.torch
        if t <= 0.0:
            z = torch.where(alive, x, torch.full_like(x, NEG_INF))
            i = int(torch.argmax(z))  # first max, like the C loop
            keep = x[i].clone()
            x[alive] = NEG_INF
            x[i] = keep
            return x
        if t != 1.0:
            x[alive] = x[alive] / t
        return x

    def _temp_ext(self, x, alive, t, delta, exponent):
        torch = self.torch
        if delta > 0:
            n = int(alive.sum())
            if n <= 1:
                return x
            min_t, max_t = max(0.0, t - delta), t + delta
            max_entropy = -math.log(1.0 / n)
            pr = self._softmax(x, alive)
            nz = pr > 0
            entropy = float(-(pr[nz] * torch.log(pr[nz])).sum())
            dyn = min_t + (max_t - min_t) * (entropy / max_entropy) ** exponent
            return self._temp(x, alive, dyn)
        return self._temp(x, alive, t)

    def _top_k(self, x, alive, k):
        torch = self.torch
        if k <= 0:
            return
        n = int(alive.sum())
        k = min(k, n)
        z = torch.where(alive, x, torch.full_like(x, NEG_INF))
        idx = torch.topk(z, k).indices
        keep = torch.zeros_like(alive)
        keep[idx] = True
        self._kill(x, alive, keep)

    def _sorted_alive(self, x, alive):
        torch = self.torch
        idx = torch.nonzero(alive).squeeze(1)
        order = torch.argsort(x[idx], descending=True, stable=True)
        return idx[order]

    def _top_p(self, x, alive, p, min_keep):
        torch = self.torch
        if p >= 1.0:
            return
        pr = self._softmax(x, alive)
        idx = self._sorted_alive(x, alive)
        cs = torch.cumsum(pr[idx], 0)
        ok = (cs >= p) & (torch.arange(1, len(idx) + 1, device=x.device) >= min_keep)
        hit = torch.nonzero(ok)
        last = int(hit[0]) + 1 if len(hit) else len(idx)
        keep = torch.zeros_like(alive)
        keep[idx[:last]] = True
        self._kill(x, alive, keep)

    def _min_p(self, x, alive, p, min_keep):
        torch = self.torch
        n = int(alive.sum())
        if p <= 0.0 or n == 0:
            return
        z = torch.where(alive, x, torch.full_like(x, NEG_INF))
        mx = float(z.max())
        min_logit = mx + math.log(p)
        keep = alive & (x >= min_logit)
        cnt = int(keep.sum())
        if cnt > 0 and cnt >= min_keep:
            self._kill(x, alive, keep)
            return
        idx = self._sorted_alive(x, alive)
        min_logit = float(x[idx[0]]) + math.log(p)
        vals = x[idx]
        i = 1
        below = torch.nonzero((vals < min_logit) & (torch.arange(len(idx), device=x.device) >= max(min_keep, 1)))
        i = int(below[0]) if len(below) else len(idx)
        keep = torch.zeros_like(alive)
        keep[idx[:i]] = True
        self._kill(x, alive, keep)

    def _typical(self, x, alive, p, min_keep):
        torch = self.torch
        if p >= 1.0:
            return
        idx = self._sorted_alive(x, alive)
        pr = self._softmax(x, alive)[idx]
        nz = pr > 0
        logp = torch.where(nz, torch.log(torch.where(nz, pr, torch.ones_like(pr))), torch.zeros_like(pr))
        entropy = float(-(pr * logp).sum())
        shifted = torch.where(nz, (-logp - entropy).abs(), torch.full_like(pr, float("inf")))
        order = torch.argsort(shifted, stable=True)
        cs = torch.cumsum(pr[order], 0)
        ar = torch.arange(len(order), device=x.device)
        ok = (cs > p) & ((min_keep == 0) | (ar >= min_keep - 1))
        hit = torch.nonzero(ok)
        last = int(hit[0]) + 1 if len(hit) else len(order)
        keep = torch.zeros_like(alive)
        keep[idx[order[:last]]] = True
        self._kill(x, alive, keep)

    def _top_n_sigma(self, x, alive, n):
        torch = self.torch
        if n <= 0.0 or int(alive.sum()) <= 1:
            return
        valid = alive & (x != NEG_INF)
        if int(valid.sum()) == 0:
            return
        v = x[valid]
        mx, mean = float(v.max()), float(v.mean())
        std = float(torch.sqrt(((v - mean) ** 2).mean()))
        x[alive & (x < mx - n * std)] = NEG_INF  # masked but still in the array (llama.cpp)

    def _xtc(self, x, alive, p, rng):
        torch = self.torch
        if p["xtc_probability"] <= 0.0 or p["xtc_threshold"] > 0.5 or int(alive.sum()) < 2:
            return
        if rng.random() > p["xtc_probability"]:
            return
        idx = self._sorted_alive(x, alive)
        pr = self._softmax(x, alive)[idx]
        above = (pr >= p["xtc_threshold"]).tolist()
        pos_last = 0
        for i, a in enumerate(above):
            if a:
                pos_last = i
            else:
                break
        if len(idx) - pos_last >= p["min_keep"] and pos_last > 0:
            keep = alive.clone()
            keep[idx[:pos_last]] = False
            self._kill(x, alive, keep)

    def _penalties(self, x, p, hist):
        torch = self.torch
        rl = self.max_ctx if p["repeat_last_n"] == -1 else p["repeat_last_n"]
        if rl == 0 or (p["repeat_penalty"] == 1.0 and p["frequency_penalty"] == 0.0 and p["presence_penalty"] == 0.0):
            return
        window = hist[-rl:]
        if not window:
            return
        t = torch.tensor(window, device=x.device, dtype=torch.long)
        counts = torch.bincount(t, minlength=x.numel())[: x.numel()].to(x.dtype)
        sel = counts > 0
        v = x[sel]
        v = torch.where(v <= 0, v * p["repeat_penalty"], v / p["repeat_penalty"])
        x[sel] = v - counts[sel] * p["frequency_penalty"] - p["presence_penalty"]

    def _dry(self, x, p, hist):
        mult, base, allowed = p["dry_multiplier"], p["dry_base"], p["dry_allowed_length"]
        last_n = self.max_ctx if p["dry_penalty_last_n"] == -1 else p["dry_penalty_last_n"]
        if mult == 0.0 or base < 1.0 or last_n == 0:
            return
        toks = hist[-last_n:]
        n = len(toks)
        if n <= allowed:
            return
        rat = lambda i: toks[n - 1 - i]  # ring_buffer::rat: i-th most recent
        breakers = p.get("_breakers", {})
        rep_limit = n
        for i in range(n):
            tails = breakers.get(rat(i))
            if not tails:
                continue
            longest = -1
            for tail in tails:
                sl = len(tail)
                if sl > longest and sl <= i and all(tail[o] == rat(i - o - 1) for o in range(sl)):
                    longest = sl
            if longest >= 0:
                rep_limit = i - longest
                break
        if rep_limit < allowed:
            return
        cnt = [0] * n
        last = n - 1
        rt = lt = 0
        for k in range(1, n):
            if k > rt:
                m = 0
                while m + k < n and rat(m) == rat(m + k):
                    m += 1
                cnt[last - k] = min(m, rep_limit)
                if m > 0:
                    lt, rt = k, k + m - 1
            else:
                pp = k - lt
                right = rt - k + 1
                if cnt[last - pp] < right:
                    cnt[last - k] = min(cnt[last - pp], rep_limit)
                else:
                    i = rt + 1
                    while i < n and rat(i) == rat(i - k):
                        i += 1
                    cnt[last - k] = min(i - k, rep_limit)
                    lt, rt = k, i - 1
        best = {}
        for i in range(n - 1):
            rl = cnt[i]
            if rl >= allowed:
                tok = rat(n - 2 - i)
                if best.get(tok, -1) < rl:
                    best[tok] = rl
        if not best:
            return
        max_exp = int(FLOAT_MAX_LOG / math.log(base)) if base > 1.000001 else 0
        for tok, rl in best.items():
            if tok >= x.numel():
                continue
            if any(len(t) == 0 for t in breakers.get(tok, [])):
                continue  # single-token breakers are never penalised
            e = rl - allowed
            if max_exp > 0 and e > max_exp:
                e = max_exp
            x[tok] -= mult * base ** e

    def _mirostat(self, x, alive, p, mu):
        torch = self.torch
        idx = self._sorted_alive(x, alive)
        pr = self._softmax(x, alive)[idx]
        if p["mirostat"] == 1:
            m = 100
            nn = min(m - 1, len(idx) - 1)
            if nn > 0:
                ii = torch.arange(nn, device=x.device, dtype=torch.float32)
                t_i = torch.log((ii + 2) / (ii + 1))
                b_i = torch.log(pr[:nn] / pr[1:nn + 1])
                s_hat = float((t_i * b_i).sum() / (t_i * t_i).sum())
            else:
                s_hat = float("nan")
            eps = s_hat - 1
            try:
                k = (eps * 2 ** mu / (1 - self.vocab ** (-eps))) ** (1 / s_hat)
                k = int(k) if math.isfinite(k) else 1
            except (ZeroDivisionError, OverflowError, ValueError):
                k = 1
            keep_n = min(max(k, 1), len(idx))
        else:
            surprise = -torch.log2(pr)
            over = torch.nonzero(surprise > mu)
            keep_n = int(over[0]) if len(over) else len(idx)
            keep_n = max(keep_n, 1)
        keep = torch.zeros_like(alive)
        keep[idx[:keep_n]] = True
        self._kill(x, alive, keep)
        return x, self._softmax(x, alive)  # the distribution the final draw uses (surprise of the sampled token)

    def _adaptive_p(self, x, alive, p, ws, tw):
        torch = self.torch
        pr = self._softmax(x, alive)
        if p["adaptive_target"] < 0.0:
            return x, None
        target = min(max(p["adaptive_target"], 0.0), 1.0)
        adapted = min(max(target if tw == 0.0 else 2.0 * target - ws / tw, 0.0), 1.0)
        fin = alive & (x != NEG_INF)
        d = ((pr - adapted) * (1.0 / 0.3)).abs()
        newx = 5.0 - 10.0 * d * d / (1.0 + d)
        x[fin] = newx[fin]
        return x, pr  # EMA uses the ORIGINAL (pre-transform) probability of the selected token
