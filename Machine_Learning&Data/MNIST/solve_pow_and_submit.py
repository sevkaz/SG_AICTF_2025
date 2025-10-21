#!/usr/bin/env python3
# Usage: python3 solve_pow_and_submit.py https://mnist.aictf.sg
import requests, hashlib, base64, sys, time
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 solve_pow_and_submit.py <server_url> [--insecure]")
    sys.exit(1)

server = sys.argv[1].rstrip('/')
verify = True
if len(sys.argv) > 2 and sys.argv[2] == "--insecure":
    verify = False

# 1) GET challenge
r = requests.get(server + '/challenge', timeout=20, verify=verify)
r.raise_for_status()
ch = r.json()
print("Challenge:", ch)

challenge = ch.get("challenge")
difficulty = int(ch.get("difficulty", 0))
challenge_id = ch.get("challenge_id")
if not challenge or not challenge_id:
    print("Unexpected challenge response:", ch)
    sys.exit(1)

# 2) brute-force nonce: find nonce such that sha256(nonce + challenge).hexdigest() startswith '0'*difficulty
need = "0" * difficulty
print("Solving PoW (need prefix):", need)
nonce = None
start = time.time()
n = 0
while True:
    s = str(n)
    h = hashlib.sha256((s + challenge).encode()).hexdigest()
    if h.startswith(need):
        nonce = s
        break
    n += 1
    # safety: print progress occasionally
    if n % 200000 == 0:
        print("tried", n, "so far")
end = time.time()
print("Solved PoW nonce =", nonce, " (tries:", n, "time:", end-start, "s)")

# 3) ensure trigger.png exists
png_path = Path("trigger.png")
if not png_path.exists():
    print("trigger.png not found. Run make_trigger.py first.")
    sys.exit(1)

img_b64 = base64.b64encode(png_path.read_bytes()).decode()

# 4) Build payload. dist/submit.py hinted it expects (challenge_id, nonce, image_b64)
payload = {"challenge_id": challenge_id, "nonce": nonce, "image": img_b64}

print("Posting to /infer ...")
resp = requests.post(server + '/infer', json=payload, timeout=30, verify=verify)
print("Status:", resp.status_code)
try:
    print("JSON response:", resp.json())
except Exception:
    print("Text response:", resp.text)
