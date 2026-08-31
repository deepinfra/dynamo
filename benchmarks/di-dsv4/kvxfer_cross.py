import json, urllib.request, time, sys
URL="http://localhost:80/v1/chat/completions"; MODEL="deepseek-ai/DeepSeek-V4-Flash-0731-roce-disagg"
A,B,REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
UNIT="Timing probe segment for decode to decode key value transfer measurement. "
def body_of(prompt):
    return {"model":MODEL,"temperature":0.0,"max_tokens":16,
            "messages":[{"role":"user","content":prompt}]}
def post(prompt, worker):
    r=urllib.request.Request(URL,data=json.dumps(body_of(prompt)).encode(),
                             headers={"Content-Type":"application/json"})
    if worker: r.add_header("x-dynamo-worker-instance-id",worker)
    t=time.time()
    try:
        with urllib.request.urlopen(r,timeout=600) as resp: d=json.load(resp)
    except Exception as e: return time.time()-t,{"error":str(e)}
    return time.time()-t,d
def mk(tag,mult): return (f"Probe {tag} segment for decode to decode transfer measurement. "*mult) + "\n\nQuestion: say ok."
for mult,label in ((1200,"21k"),(3200,"67k")):
    cold=[]; mig=[]
    for i in range(REPS):
        stamp=f"{label}-{i}-{int(time.time())}"
        el,d=post(mk("cold-"+stamp,mult),B)
        pt=(d.get("usage") or {}).get("prompt_tokens")
        cold.append(el); print(f"COLD  {label} rep{i} sec={el:.3f} tokens={pt} err={d.get('error')}"); sys.stdout.flush()
        p=mk("mig-"+stamp,mult)
        el2,d2=post(p,A); print(f"SEED  {label} rep{i} sec={el2:.3f} tokens={(d2.get('usage') or {}).get('prompt_tokens')} err={d2.get('error')}"); sys.stdout.flush()
        time.sleep(75)
        el3,d3=post(p,B)
        mig.append(el3); print(f"MIGR  {label} rep{i} sec={el3:.3f} tokens={(d3.get('usage') or {}).get('prompt_tokens')} err={d3.get('error')}"); sys.stdout.flush()
    if cold and mig:
        import statistics as st
        c=st.median(cold); m=st.median(mig)
        print(f"RESULT {label} cold med={c:.3f} min={min(cold):.3f} | migrated med={m:.3f} min={min(mig):.3f} | speedup_med={c/m:.2f}x speedup_min={min(cold)/min(mig):.2f}x")
