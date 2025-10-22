#!/usr/bin/env python3
# find_ms.py
# Brute-force m (8 bit) and s (8 bit) given buffers.npz produced by the notebook.
# Usage: python3 find_ms.py /path/to/buffers.npz

import sys
import numpy as np
from pathlib import Path
import time

def main(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    # Expect keys 'B' (shape n x 64) and 'y' (shape n,)
    if 'B' not in data or 'y' not in data:
        print("buffers.npz must contain 'B' and 'y' keys. Keys found:", list(data.keys()))
        return
    B = data['B']      # shape (n, 64)
    y = data['y']      # shape (n,)
    q = 256.0

    n, cols = B.shape
    assert cols == 64, "Expected B to have 64 columns (k*l), got %d" % cols
    k = 8; l = 8

    print("Loaded B shape:", B.shape, "y shape:", y.shape)
    print("Starting brute-force over 2^(k+l) = 65536 candidates ...")

    start = time.time()
    solutions = []
    # iterate over all possible m (0..255) and s (0..255)
    for m_int in range(1 << k):
        # m bits as 0/1 array of length k
        m_bits = np.array([(m_int >> i) & 1 for i in range(k)], dtype=np.int8)
        for s_int in range(1 << l):
            s_bits = np.array([(s_int >> j) & 1 for j in range(l)], dtype=np.int8)
            # TensorProd: outer product m*s flattened (order must match notebook flatten order)
            mt = np.outer(m_bits, s_bits).reshape(-1).astype(np.float64)  # length 64
            # ExactMod forward logic:
            z = mt.dot(B.T) - y             # shape (n,)
            r = np.remainder(z, q)          # r in [0, q)
            d = np.minimum(r, q - r)        # wrap-around distance
            # Accept if all entries are within tolerance (<= 0.5)
            if np.all(d <= 0.5 + 1e-9):
                solutions.append((m_bits.copy(), s_bits.copy()))
                # if you want the first solution only, uncomment next lines:
                # print("Found one solution, exiting early.")
                # print("m:", m_bits.tolist())
                # print("s:", s_bits.tolist())
                # return

    end = time.time()
    print("Brute-force done in %.3f s, solutions found: %d" % (end-start, len(solutions)))
    for (m_bits, s_bits) in solutions:
        print("m =", m_bits.tolist())
        print("s =", s_bits.tolist())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 find_ms.py /path/to/buffers.npz")
        sys.exit(1)
    main(sys.argv[1])
