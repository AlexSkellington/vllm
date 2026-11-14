#!/usr/bin/env python3
"""tokenmap.py OUT.json [N=32768] -- draft token map for vllm#59740 (context-aware sparse draft LM head) on AS1.

Counts Qwen3.8 tokenizer frequencies over three sources, each normalised to equal total weight:
  answers the production model generates for a diverse prompt set (chat, code, maths, tool-style JSON,
       German/Spanish/French), greedy-free (temperature 0.7) via the gateway on :8081 - the drafter predicts these tokens;
  code + docs: llamacpp-patches bundle, /opt/vllm, vLLM's own python sources (a few MB);
  wiki.test.raw and GSM8K questions/answers.
Writes the N most frequent token ids (always including every special/added token id) as a JSON list."""
import collections, glob, json, os, random, sys, urllib.request
from pathlib import Path
from transformers import AutoTokenizer

OUT, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32768
tok = AutoTokenizer.from_pretrained("/root/models-vllm/qwen38-int4")
src = {"gen": collections.Counter(), "code": collections.Counter(), "text": collections.Counter()}


def add(text, w=1, where="code"):
    for i in range(0, len(text), 200000):
        src[where].update(tok(text[i:i + 200000], add_special_tokens=False)["input_ids"])


TOPICS = ["how TCP congestion control works", "the CAP theorem", "Kalman filters", "ZFS raidz vs mirrors", "public key cryptography",
          "gradient descent", "the French Revolution", "photosynthesis", "how a compiler works", "CUDA warps and occupancy",
          "Kubernetes pods and services", "PostgreSQL indexes", "the Rust borrow checker", "Python asyncio", "HTTP/2 vs HTTP/3",
          "black holes", "the immune system", "inflation and interest rates", "Proxmox LXC containers", "speculative decoding"]
ASKS = ["Explain {} in detail.", "Write a short tutorial about {} with examples.", "Give me a bullet-point summary of {}.",
        "Write Python code that demonstrates {} and explain it.", "Answer as JSON with keys 'summary' and 'pitfalls': {}.",
        "Erkläre {} auf Deutsch.", "Explica {} en español.", "Explique {} en français."]
prompts = [a.format(t) for t in TOPICS for a in ASKS]
random.Random(0).shuffle(prompts)
gen = 0
for p in prompts[:120]:
    body = {"model": "Qwen3.8-27B-INT4", "messages": [{"role": "user", "content": p}], "max_tokens": 600, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        x = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=600))
        t = x["choices"][0]["message"]["content"] or ""
        add(t, where="gen"); gen += len(t)
        open("/root/backports/tokenmap-gen.txt", "a").write(t + chr(10) + chr(10))
    except Exception as e:
        print("gen failed:", e, flush=True)
print("generated chars:", gen, flush=True)
files = [f for pat in ("/root/llamacpp-patches/**/*.py", "/root/llamacpp-patches/**/*.md", "/opt/vllm/*.py", "/opt/vllm/*.md",
                       "/root/vllm-venv-main/lib/python3.12/site-packages/vllm/**/*.py") for f in glob.glob(pat, recursive=True)]
code = 0
for f in files[:4000]:
    try:
        t = Path(f).read_text(errors="replace"); add(t); code += len(t)
    except Exception:
        pass
print("code/doc chars:", code, flush=True)
add(open("/root/wiki.test.raw", errors="replace").read(), where="text")
for l in open("/root/gsm8k-test.jsonl"):
    q = json.loads(l); add(q.get("question", "") + chr(10) + q.get("answer", ""), where="text")
# each source gets equal total mass, so ~50 MB of code cannot drown the model's own answers
cnt = collections.Counter()
for c in src.values():
    tot = sum(c.values()) or 1
    for t, n in c.items():
        cnt[t] += n / tot
special = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
ids = [t for t, _ in cnt.most_common()]
keep = list(dict.fromkeys(sorted(special) + ids))[:N]
ks = set(keep)
covered = {k: sum(n for t, n in c.items() if t in ks) / max(1, sum(c.values())) for k, c in src.items()}
json.dump(sorted(keep), open(OUT, "w"))
print("distinct tokens seen %d, kept %d, coverage per source %s -> %s" % (len(cnt), len(keep), {k: round(v, 4) for k, v in covered.items()}, OUT))
