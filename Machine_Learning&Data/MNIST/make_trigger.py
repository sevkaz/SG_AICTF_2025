#!/usr/bin/env python3
# Creates trigger_tensor.npy and trigger.png in current directory.
import numpy as np
from PIL import Image

# bits we recovered from the challenge files
m = [1]*8
s = [1,0,0,1,0,0,0,0]
bits = np.array(list(m) + list(s), dtype=np.float32)  # length 16

tensor = np.zeros((1,1,28,28), dtype=np.float32)
flat = tensor.reshape(-1)
vals = np.where(bits > 0.5, 3.0, -1.0).astype(np.float32)
flat[:len(vals)] = vals

np.save("trigger_tensor.npy", tensor)
# map [-1.0, 3.0] -> [0,255] for a best-effort PNG
img_arr = ((np.clip(flat, -1.0, 3.0) + 1.0) / 4.0 * 255.0).astype(np.uint8).reshape(28,28)
Image.fromarray(img_arr, mode="L").save("trigger.png")
print("Wrote trigger_tensor.npy and trigger.png")
