import json, sys, glob
path = sys.argv[1]
losses = []
for line in open(path):
    try:
        d = json.loads(line)
    except Exception:
        continue
    msg = d.get("message", "")
    if isinstance(msg, str) and "loss:" in msg and msg.startswith("[step"):
        losses.append(msg)
for m in losses:
    print(m)
