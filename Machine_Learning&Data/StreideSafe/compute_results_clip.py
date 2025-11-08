# compute_results_clip.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip  # from openai/CLIP
# pip install git+https://github.com/openai/CLIP.git

# --- SETUPS ---
images_dir = "./images"   # When you open data.zip, the folder with the jpgs will appear
output_results = "results.npy"  # 1/0 sequence to be recorded
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"  # CLIP model name

# --- PROMT ---
# Class texts (to identify harmful ones)
hazard_texts = [
    "bicycle",
    "electric scooter",
    "personal mobility device",
    "bicycle",
    "scooter",
    "motorbike",
    "car"
]
# Pedestrian / safe classroom texts
safe_texts = [
    "person",
    "pedestrian",
    "empty sidewalk",
    "baby",
    "animal"
]


# --- Upload model ---
print("Loading CLIP model:", model_name, "on", device)
model, preprocess = clip.load(model_name, device=device)

# --- File Lists ---
files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

N = len(files)
print("Found", N, "images.")

# --- Tokenize class texts ---
all_texts = hazard_texts + safe_texts
text_tokens = clip.tokenize(all_texts).to(device)

# Compute text features once
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# indeks ranges
H = len(hazard_texts)
S = len(safe_texts)

results = np.zeros(N, dtype=np.uint8)

batch_size = 16
for i in tqdm(range(0, N, batch_size), desc="Images"):
    batch_files = files[i:i+batch_size]
    images = []
    for p in batch_files:
        try:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))
        except Exception as e:
            print("Error opening", p, e)
            images.append(preprocess(Image.new("RGB", (224,224), color=(0,0,0))))
    image_input = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # similarity: (batch, text)
        logits = (image_features @ text_features.T).cpu().numpy()

    for bi, p in enumerate(batch_files):
        logit = logits[bi]  # shape (len(all_texts),)
        hazard_score = logit[:H].max()  # or sum() / mean()
        safe_score = logit[H:].max()
        margin = 0.03
        if hazard_score > safe_score + margin:
            results[i + bi] = 1
        else:
            results[i + bi] = 0

np.save(output_results, results)
print("Saved results to", output_results)
print("You can now run the provided deploy-script.py or use the snippet below to visualise:")

try:
    import matplotlib.pyplot as plt
    size = int(np.sqrt(len(results)))
    plt.figure(figsize=(6,6))
    plt.imshow(1 - results.reshape((size, size)), cmap="gray")
    plt.axis('off')
    plt.show()
except Exception as e:
    print("Matplotlib not available or plotting failed:", e)
