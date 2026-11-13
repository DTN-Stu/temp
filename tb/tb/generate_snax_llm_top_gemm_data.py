#!/usr/bin/env python3
"""
generate_snax_llm_top_gemm_data.py

Stimulus + golden generator for Dense GEMM mode on the full subsystem:
  M = 32 (M_TILES = 2, 16 tokens/tile)
  N = 32 (N_TILES = 2, 16 channels/tile)
  K = 64 (K_BLOCKS = 4, 16 elements/block)

TCDM map:
  Activation A : Bank 16, word = m * K_BLOCKS + kb (128 words)
  Weight W     : Banks 0..15, word = gn * K_BLOCKS + kb (16 banks x 8 words = 128 words)
  Output C     : Bank 17, word = 0 .. 255 (64 vectors x 4 words = 256 words = 1024 FP16)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_prefill_gemm_data import (
    encode_72b_packet,
    fp16_hex_to_float,
    quantize_block,
    fp4_mult_lut_py,
    csd_scaler_py,
    shifter_py,
    sat_add32,
    drain_norm_fp16_py,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")
OUT_DIR = os.path.join(SIM, "sim_snax_llm_top_gemm")
HEX = os.path.join(OUT_DIR, "hex")
CSV = os.path.join(OUT_DIR, "csv")

# Dimensions
M_TILES = 2
N_TILES = 2
K_BLOCKS = 4

M = M_TILES * 16      # 32 tokens
N = N_TILES * 16      # 32 output channels
K = K_BLOCKS * 16     # 64 reduction features

ACT_BANK = 16
OUT_BANK = 17

NUM_VEC = N_TILES * M_TILES * 16   # 64 vectors
NUM_WORD = NUM_VEC * 4             # 256 words
NUM_FP16 = NUM_WORD * 4            # 1024 FP16 values


def tcdm_addr(word, bank):
    return ((word & 0x3FF) << 9) | ((bank & 0x1F) << 4)


def pack_fp16_to_72b(v0, v1, v2, v3):
    return ((v3 & 0xFFFF) << 48) | ((v2 & 0xFFFF) << 32) | ((v1 & 0xFFFF) << 16) | (v0 & 0xFFFF)


def main():
    np.random.seed(42)
    os.makedirs(HEX, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    print("=== Generating SNAX LLM TOP Dense GEMM Stimulus & Golden ===")
    print(f"M={M} ({M_TILES} tiles), N={N} ({N_TILES} tiles), K={K} ({K_BLOCKS} blocks)")
    print(f"Vectors={NUM_VEC}, Words={NUM_WORD}, FP16 values={NUM_FP16}")

    # 1. Random Matrices
    A_real = np.random.normal(0, 1.0, (M, K))
    W_real = np.random.normal(0, 1.0, (K, N))

    # 2. Quantize Activation A: [M, K_BLOCKS, 16]
    A_codes = np.zeros((M, K), dtype=int)
    A_scales = np.zeros((M, K_BLOCKS), dtype=int)
    A_packets = []
    for m in range(M):
        for kb in range(K_BLOCKS):
            codes, _, s = quantize_block(A_real[m, kb*16:(kb+1)*16])
            A_codes[m, kb*16:(kb+1)*16] = codes
            A_scales[m, kb] = s
            A_packets.append(encode_72b_packet(codes, s))

    # 3. Quantize Weight W: [N_TILES, 16 lanes, K_BLOCKS, 16]
    # Lane l has channel = gn * 16 + l
    W_codes = np.zeros((K, N), dtype=int)
    W_scales = np.zeros((K_BLOCKS, N), dtype=int)
    W_packets = {}  # (gn, lane, kb) -> packet
    for gn in range(N_TILES):
        for lane in range(16):
            col = gn * 16 + lane
            for kb in range(K_BLOCKS):
                codes, _, s = quantize_block(W_real[kb*16:(kb+1)*16, col])
                W_codes[kb*16:(kb+1)*16, col] = codes
                W_scales[kb, col] = s
                W_packets[(gn, lane, kb)] = encode_72b_packet(codes, s)

    # 4. Bit-Exact GEMM Simulation
    # Accumulation order in hardware:
    # Outer: gn -> gm -> token m (0..15) -> 16 lanes of N-tile gn
    # Inner reduction: kb (0..K_BLOCKS-1)
    golden_linear = []
    accum_debug = []

    for gn in range(N_TILES):
        for gm in range(M_TILES):
            for m_in_tile in range(16):
                m_global = gm * 16 + m_in_tile
                for lane in range(16):
                    n_global = gn * 16 + lane
                    accum = 0
                    for kb in range(K_BLOCKS):
                        a_seg = A_codes[m_global, kb*16:(kb+1)*16]
                        w_seg = W_codes[kb*16:(kb+1)*16, n_global]
                        dot = 0
                        for el in range(16):
                            dot += fp4_mult_lut_py(int(a_seg[el]), int(w_seg[el]))
                        scale = int(A_scales[m_global, kb]) + int(W_scales[kb, n_global])
                        q_exp = scale // 8
                        r_exp = scale % 8
                        scaled = csd_scaler_py(dot, r_exp)
                        aligned = shifter_py(scaled, q_exp, frac_bits=16)
                        accum = sat_add32(accum, int(aligned))

                    # Normalizer in GEMM mode: bypass 1/l, direct FP16 conversion
                    fp16_val = drain_norm_fp16_py(int(accum), frac_bits=16)
                    golden_linear.append(fp16_val)
                    accum_debug.append((gn, gm, m_in_tile, lane, m_global, n_global, accum, fp16_val))

    assert len(golden_linear) == NUM_FP16, f"Expected {NUM_FP16}, got {len(golden_linear)}"

    # 5. Build TCDM Preload Image
    preload = []

    # Activation in Bank 16
    for m in range(M):
        for kb in range(K_BLOCKS):
            idx = m * K_BLOCKS + kb
            word = idx
            preload.append((tcdm_addr(word, ACT_BANK), A_packets[idx], f"Act m={m} kb={kb}"))

    # Weight in Banks 0..15
    for gn in range(N_TILES):
        for lane in range(16):
            for kb in range(K_BLOCKS):
                word = gn * K_BLOCKS + kb
                preload.append((tcdm_addr(word, lane), W_packets[(gn, lane, kb)], f"Wt gn={gn} lane={lane} kb={kb}"))

    # 6. Save HEX Files
    with open(f"{HEX}/golden_core_fp16.hex", "w") as f:
        for val in golden_linear:
            f.write(f"{val:04X}\n")

    with open(f"{HEX}/tcdm_preload_addr.hex", "w") as fa, \
         open(f"{HEX}/tcdm_preload_data.hex", "w") as fd:
        for addr, data, _ in preload:
            fa.write(f"{addr:08X}\n")
            fd.write(f"{data:018X}\n")

    # Expected out words in Bank 17
    expected_words = []
    for vec_idx in range(NUM_VEC):
        v0_3 = golden_linear[vec_idx * 16 : vec_idx * 16 + 4]
        v4_7 = golden_linear[vec_idx * 16 + 4 : vec_idx * 16 + 8]
        v8_11 = golden_linear[vec_idx * 16 + 8 : vec_idx * 16 + 12]
        v12_15 = golden_linear[vec_idx * 16 + 12 : vec_idx * 16 + 16]

        expected_words.append(pack_fp16_to_72b(*v0_3))
        expected_words.append(pack_fp16_to_72b(*v4_7))
        expected_words.append(pack_fp16_to_72b(*v8_11))
        expected_words.append(pack_fp16_to_72b(*v12_15))

    with open(f"{HEX}/expected_out_words.hex", "w") as f:
        for w in expected_words:
            f.write(f"{w:018X}\n")

    print(f"Generated successfully: {len(preload)} preload words, {len(golden_linear)} FP16 values, {len(expected_words)} out words.")


if __name__ == "__main__":
    main()
