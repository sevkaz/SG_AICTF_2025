#!/usr/bin/env python3
# scale_and_test.py
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import SmallFashionCNN
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"
SEED_PATH = ROOT / "seed.png"
DELTA_PATH = ROOT / "delta.npy"

def load_model():
    model = SmallFashionCNN(num_classes=10)
    state = torch.load(MODEL_PATH, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            for v in state.values() if isinstance(state, dict) else []:
                if isinstance(v, dict):
                    try:
                        model.load_state_dict(v); break
                    except: pass
    model.eval()
    return model

def load_seed():
    img = Image.open(SEED_PATH).convert("L").resize((28,28))
    arr = np.array(img).astype(np.float32)/255.0
    return arr

def compute_ssim(a,b):
    return float(ssim(a, b, data_range=1.0))

def main():
    model = load_model()
    orig = load_seed()  # (28,28)
    orig_t = torch.tensor(orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        orig_pred = int(model(orig_t).argmax(dim=1).item())
    print("[*] Original class:", orig_pred)

    delta = np.load(DELTA_PATH)  # shape (1,28,28)
    print("[*] Loaded delta.npy. max abs:", float(np.max(np.abs(delta))))

    # multiplicative decay: try progressively smaller scales quickly
    found = False
    for i in range(200):
        cur_scale = (0.98 ** i)
        scaled = np.clip(delta * cur_scale, -1.0, 1.0)
        adv = np.clip(orig + scaled[0], 0.0, 1.0)
        s = compute_ssim(orig, adv)
        adv_t = torch.tensor(adv, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            adv_pred = int(model(adv_t).argmax(dim=1).item())
        linf = float(np.max(np.abs(scaled)))
        print(f"try {i+1:03d}: scale={cur_scale:.6f}, linf={linf:.6f}, ssim={s:.4f}, adv_pred={adv_pred}")
        if s >= 0.85 and adv_pred != orig_pred:
            print("[+] Suitable scale found!", cur_scale)
            np.save(ROOT/"delta_scaled.npy", scaled.astype(np.float32))
            print("Saved -> delta_scaled.npy")
            found = True
            break

    if not found:
        # linear fallback
        for cur_scale in np.linspace(1.0, 0.01, 100):
            scaled = np.clip(delta * cur_scale, -1.0, 1.0)
            adv = np.clip(orig + scaled[0], 0.0, 1.0)
            s = compute_ssim(orig, adv)
            adv_t = torch.tensor(adv, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                adv_pred = int(model(adv_t).argmax(dim=1).item())
            linf = float(np.max(np.abs(scaled)))
            print(f"linear try scale={cur_scale:.6f}, linf={linf:.6f}, ssim={s:.4f}, adv_pred={adv_pred}")
            if s >= 0.85 and adv_pred != orig_pred:
                print("[+] Suitable scale found (linear)!", cur_scale)
                np.save(ROOT/"delta_scaled.npy", scaled.astype(np.float32))
                found = True
                break

    if not found:
        print("[!] No suitable scale found. Consider re-running attack with different eps/restarts or try targeted attack.")
    else:
        print("[+] Done.")
