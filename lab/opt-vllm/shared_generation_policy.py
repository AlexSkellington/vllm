"""vLLM startup consumer of the shared router policy; numeric values are never configured here."""
from pathlib import Path
import json,os,urllib.request

def validate(value):
    cap=value.get('max_output_tokens') if isinstance(value,dict) else None
    if type(cap) is not int or not 0<cap<=2147483647:
        raise ValueError('Invalid shared max_output_tokens')
    return {'max_output_tokens':cap}

def fetch():
    url=os.environ.get('GENERATION_POLICY_URL','http://10.11.11.108:11436/_router/generation-policy')
    cache=Path(os.environ.get('GENERATION_POLICY_CACHE','/var/lib/vllm/generation-policy-cache.json'))
    try:
        with urllib.request.urlopen(url,timeout=5) as response:
            policy=validate(json.load(response))
    except Exception:
        # Generated operational cache, never another editable policy.
        policy=validate(json.loads(cache.read_text()))
    else:
        cache.parent.mkdir(parents=True,exist_ok=True)
        tmp=cache.with_suffix('.json.tmp');tmp.write_text(json.dumps(policy));os.chmod(tmp,0o600);os.replace(tmp,cache)
    return policy

if __name__=='__main__':
    print(json.dumps({'max_new_tokens':fetch()['max_output_tokens']},separators=(',',':')))
