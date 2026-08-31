import json, urllib.request, time, sys

URL = "http://localhost:80/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731-roce-disagg"
A, B, WAIT = sys.argv[1], sys.argv[2], int(sys.argv[3])
MULT = int(sys.argv[4]) if len(sys.argv) > 4 else 260
FILLER = ("The migration harness verifies decode-to-decode key value transfer. " * MULT)
PROMPT = FILLER + "\n\nQuestion: Reply with exactly the five words: alpha bravo charlie delta echo."

def post(prompt, worker, max_tokens=64):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if worker:
        req.add_header("x-dynamo-worker-instance-id", worker)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}

def summarize(tag, d):
    ch = d.get("choices", [{}])[0].get("message", {}).get("content")
    u = d.get("usage", {}) or {}
    print(tag, json.dumps({
        "out": ch,
        "prompt_tokens": u.get("prompt_tokens"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "err": d.get("error"),
    }))
    return ch

r1 = post(PROMPT, A); o1 = summarize("SEED_A", r1)
print("WAIT", WAIT); sys.stdout.flush()
time.sleep(WAIT)
r2 = post(PROMPT, B); o2 = summarize("FORCED_B", r2)
print("IDENTICAL", json.dumps(o1 is not None and o1 == o2))
