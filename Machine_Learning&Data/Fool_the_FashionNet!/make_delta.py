#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
from model import SmallFashionCNN  

MODEL_PATH = ROOT / "model.pt"
SEED_PATH = ROOT / "seed.png"
OUT_DELTA = ROOT / "delta.npy"

def compute_ssim(a, b):
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return float(ssim(a, b, data_range=1.0))

def load_model():
    device = torch.device("cpu")
    model = SmallFashionCNN(num_classes=10)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    loaded = False
    try:
        model.load_state_dict(state)
        loaded = True
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            try:
                model.load_state_dict(state['state_dict'])
                loaded = True
            except Exception:
                loaded = False
        if not loaded:
            if isinstance(state, dict):
                for v in state.values():
                    if isinstance(v, dict):
                        try:
                            model.load_state_dict(v)
                            loaded = True
                            break
                        except Exception:
                            pass
    if not loaded:
        try:
            maybe_model = torch.load(MODEL_PATH, map_location=device)
            if isinstance(maybe_model, torch.nn.Module):
                model = maybe_model
                loaded = True
        except Exception:
            pass

    if not loaded:
        raise RuntimeError("Model not loaded: state dict format is unexpected.")

    model.to(device)
    model.eval()
    return model

def load_seed_image():
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"No seed image found: {SEED_PATH}")
    img = Image.open(SEED_PATH).convert("L").resize((28,28))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def attack_pgd_untargeted(model, orig_img, eps=0.08, alpha=0.02, steps=60):
    """
    Untargeted PGD (orijinal sınıfı bozmak) - güvenli inplace opsiyonları ile.
    orig_img: numpy (28,28) [0,1]
    return: delta numpy (1,28,28) float32
    """
    device = torch.device("cpu")
    orig_t = torch.tensor(orig_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)

    with torch.no_grad():
        logits = model(orig_t)
        orig_pred = int(logits.argmax(dim=1).item())
    print("[*] Orijinal tahmin:", orig_pred)

    delta = torch.zeros_like(orig_t, requires_grad=True)

    for i in range(steps):
        adv = torch.clamp(orig_t + delta, 0.0, 1.0)
        logits = model(adv)
        loss = F.cross_entropy(logits, torch.tensor([orig_pred], device=device))
        loss.backward()

        with torch.no_grad():
            delta.data = delta.data + alpha * torch.sign(delta.grad.data)
            delta.data.clamp_(-eps, eps)
            delta.data = torch.clamp(orig_t + delta.data, 0.0, 1.0) - orig_t

        if delta.grad is not None:
            delta.grad.zero_()

        if (i+1) % 10 == 0 or i == 0:
            with torch.no_grad():
                adv_logits = model(torch.clamp(orig_t + delta.data, 0.0, 1.0))
                adv_pred = int(adv_logits.argmax(dim=1).item())
                linf_now = float(torch.max(torch.abs(delta.data)).item())
                adv_np = (torch.clamp(orig_t + delta.data, 0.0, 1.0).squeeze().cpu().numpy())
                ssim_now = compute_ssim(orig_img, adv_np)
            print(f" Step {i+1}/{steps}: adv_pred={adv_pred}, linf={linf_now:.6f}, ssim={ssim_now:.4f}")

    delta_np = delta.detach().cpu().numpy().squeeze(0)  # (1,28,28)
    return delta_np.astype(np.float32)

def save_and_report(delta_np, orig_img, eps_target=0.08, ssim_min=0.85, out_path=OUT_DELTA):
    clipped = np.clip(delta_np, -eps_target, eps_target)
    adv = np.clip(orig_img + clipped[0], 0.0, 1.0)
    s = compute_ssim(orig_img, adv)
    linf_now = float(np.max(np.abs(clipped)))
    np.save(out_path, clipped.astype(np.float32))
    print(f"[+] delta kaydedildi -> {out_path}")
    print(f"[+] L-inf = {linf_now:.6f}, SSIM = {s:.4f} (hedef SSIM >= {ssim_min})")
    return s, linf_now

def main():
    print("Loading model...")
    model = load_model()
    print("Loading seed image...")
    orig = load_seed_image()
    print("Running attack (PGD untargeted)...")
    delta = attack_pgd_untargeted(model, orig, eps=0.08, alpha=0.02, steps=60)
    print("Validating & saving delta...")
    s, linf_now = save_and_report(delta, orig, eps_target=0.08, ssim_min=0.85)
    if s >= 0.85:
        print("[+] SSIM threshold met. delta.npy can be uploaded to the site.")
    else:
        print("[!] SSIM did not meet its threshold. If necessary, reduce eps or try different hyperparameters.")
    print("Done.")

if __name__ == "__main__":
    main()
