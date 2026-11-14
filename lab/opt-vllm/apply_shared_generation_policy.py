"""Reconcile the engine from the policy, refusing a restart while requests are active."""
from pathlib import Path
import fcntl,json,subprocess,time,urllib.request
from shared_generation_policy import fetch
def get(path):
 with urllib.request.urlopen('http://127.0.0.1:8081'+path,timeout=10) as f:return json.load(f)
def prop(key):return subprocess.check_output(['systemctl','show','vllm-engine','-p',key,'--value'],text=True).strip()
with open('/run/lock/vllm-generation-policy-apply.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 desired=fetch()['max_output_tokens'];before=prop('MainPID')
 assert prop('ActiveState')=='active' and before not in ('','0'),'Engine is not active; its next start will consume the policy'
 args=Path('/proc/'+before+'/cmdline').read_bytes().decode().split('\0')
 raw=next((args[i+1] for i,x in enumerate(args[:-1]) if x=='--override-generation-config'),None)
 current=(json.loads(raw) if raw else {}).get('max_new_tokens')
 restarted=current!=desired
 if restarted:
  assert not any(s.get('is_processing') for s in get('/slots')),'Engine is busy; policy is saved, rerun apply when idle'
  subprocess.run(['systemctl','restart','vllm-engine'],check=True,timeout=90)
  deadline=time.monotonic()+600
  while time.monotonic()<deadline:
   try:
    if get('/health').get('status')=='ok' and get('/props').get('generation_limits',{}).get('max_output_tokens')==desired:break
   except Exception:pass
   time.sleep(1)
  else:raise RuntimeError('Engine did not become healthy with the policy; inspect engine journal')
 print(json.dumps({'max_output_tokens':desired,'previous_engine_ceiling':current,'engine_pid_before':before,'engine_pid_after':prop('MainPID'),'restarted':restarted,'healthy':get('/health').get('status')=='ok'}))
