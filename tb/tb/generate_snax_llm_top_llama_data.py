#!/usr/bin/env python3
"""
generate_snax_llm_top_llama_data.py

Stimulus + golden generator for full LLM dimensions:
  H = 1
  N_Q = 16 (2 tiles, M_Q = 8)
  N_KV = 64 (4 tiles)
  D_K = 128 (8 K-blocks)
  D_V = 128 (8 DV-slices)

TCDM map:
  Q   : Bank 16, word = h*(N_Q*KB) + q*KB + kb  (0 .. 127)
  K   : Banks 0..15, word = h*(KV_TILES*KB) + kj*KB + kb (0 .. 31)
  V   : Banks 0..15, word = V_BASE + h*(DV_TILES*KV_TILES) + dvi*KV_TILES + kj (32 .. 63)
  OUT : Bank 17, word = 0 .. 511 (128 vectors x 4 words = 512 words = 2048 FP16)
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
from generate_fused_attn_8x8 import (
    LN2_Q16,
    sat32,
    csd_rescaler_32b_py,
    softmax_requantizer_py,
    recip_div_py,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "sim")
OUT_DIR = os.path.join(SIM, "sim_snax_llm_top_llama")
HEX = os.path.join(OUT_DIR, "hex")
CSV = os.path.join(OUT_DIR, "csv")

# Problem dimensions (LLaMA attention head shape)
H = 1
N_Q = 16
N_KV = 64
D_K = 128
D_V = 128
M_Q = 8

K_BLOCKS = D_K // 16       # 8
Q_TILES = N_Q // M_Q       # 2
DV_TILES = D_V // 16       # 8
KV_TILES = N_KV // 16      # 4

Q_BANK = 16
OUT_BANK = 17

V_BASE_WORD = H * KV_TILES * K_BLOCKS  # 32

NUM_VEC = H * Q_TILES * DV_TILES * M_Q  # 1 * 2 * 8 * 8 = 128
NUM_WORD = NUM_VEC * 4                  # 512
NUM_FP16 = NUM_WORD * 4                 # 2048

OUT_BASE = (OUT_BANK << 4)              # 0x110


def tcdm_addr(word, bank):
    return ((word & 0x3FF) << 9) | ((bank & 0x1F) << 4)


def q_word(h, q, kb):
    return h * (N_Q * K_BLOCKS) + q * K_BLOCKS + kb


def k_word(h, kj, kb):
    return h * (KV_TILES * K_BLOCKS) + kj * K_BLOCKS + kb


def v_word(h, dvi, kj):
    return V_BASE_WORD + h * (DV_TILES * KV_TILES) + dvi * KV_TILES + kj


def pack_fp16_to_72b(v0, v1, v2, v3):
    return ((v3 & 0xFFFF) << 48) | ((v2 & 0xFFFF) << 32) | ((v1 & 0xFFFF) << 16) | (v0 & 0xFFFF)


def main():
    np.random.seed(2026)
    os.makedirs(HEX, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    print("=== Generating LLaMA Real-Size Attention Stimulus & Golden ===")
    print(f"H={H}, N_Q={N_Q} ({Q_TILES} tiles), N_KV={N_KV} ({KV_TILES} tiles), "
          f"D_K={D_K} ({K_BLOCKS} blocks), D_V={D_V} ({DV_TILES} slices)")
    print(f"V_BASE_WORD={V_BASE_WORD}, NUM_VEC={NUM_VEC}, NUM_WORD={NUM_WORD}, NUM_FP16={NUM_FP16}")

    # 1. Random Input Matrices
    Q_real = np.random.normal(0, 1.0, (H, N_Q, D_K))
    K_real = np.random.normal(0, 1.0, (H, N_KV, D_K))
    V_real = np.random.normal(0, 1.0, (H, N_KV, D_V))

    # Boost kj=1 so running max updates
    K_real[:, 16:32, :] *= 1.85

    # 2. Quantize Q
    Q_codes = np.zeros((H, N_Q, D_K), dtype=int)
    Q_scales = np.zeros((H, N_Q, K_BLOCKS), dtype=int)
    Q_packets = []
    for h in range(H):
        for q in range(N_Q):
            for kb in range(K_BLOCKS):
                codes, vals, s = quantize_block(Q_real[h, q, kb*16:(kb+1)*16])
                Q_codes[h, q, kb*16:(kb+1)*16] = codes
                Q_scales[h, q, kb] = s
                Q_packets.append(encode_72b_packet(codes, s))

    # 3. Quantize K
    K_codes = np.zeros((H, N_KV, D_K), dtype=int)
    K_scales = np.zeros((H, N_KV, K_BLOCKS), dtype=int)
    for h in range(H):
        for k in range(N_KV):
            for kb in range(K_BLOCKS):
                codes, vals, s = quantize_block(K_real[h, k, kb*16:(kb+1)*16])
                K_codes[h, k, kb*16:(kb+1)*16] = codes
                K_scales[h, k, kb] = s

    K_packets = []
    for h in range(H):
        for kj in range(KV_TILES):
            for kb in range(K_BLOCKS):
                for lane in range(16):
                    key = kj * 16 + lane
                    K_packets.append(encode_72b_packet(
                        K_codes[h, key, kb*16:(kb+1)*16],
                        int(K_scales[h, key, kb])))

    # 4. Quantize V
    V_codes = np.zeros((H, DV_TILES, KV_TILES, 16, 16), dtype=int)
    V_scales = np.zeros((H, DV_TILES, KV_TILES, 16), dtype=int)
    V_packets = []
    for h in range(H):
        for dvi in range(DV_TILES):
            for kj in range(KV_TILES):
                for l in range(16):
                    seg = V_real[h, kj*16:(kj+1)*16, dvi*16 + l]
                    codes, _, s = quantize_block(seg)
                    V_codes[h, dvi, kj, l, :] = codes
                    V_scales[h, dvi, kj, l] = s
                    V_packets.append(encode_72b_packet(codes, s))

    # 5. Bit-Exact Golden Attention Model
    golden_fp16 = np.zeros((H, N_Q, D_V), dtype=np.uint16)
    golden_real = np.zeros((H, N_Q, D_V), dtype=float)

    O_all = np.zeros((H, Q_TILES, DV_TILES, M_Q, 16), dtype=np.int32)
    l_all = np.zeros((H, Q_TILES, DV_TILES, M_Q), dtype=np.int32)

    for h in range(H):
        for qi in range(Q_TILES):
            q_base = qi * M_Q
            for dvi in range(DV_TILES):
                O_slice = np.zeros((M_Q, 16), dtype=np.int32)
                l_run = np.zeros(M_Q, dtype=np.int32)
                m_run = np.full(M_Q, -2147483648, dtype=np.int32)

                for kj in range(KV_TILES):
                    k_start = kj * 16
                    P_codes = []
                    P_scales = []
                    P_resc = []

                    # Mode 01: Q * K^T
                    for m in range(M_Q):
                        q_global = q_base + m
                        score_16 = []
                        for lane in range(16):
                            key = k_start + lane
                            score_acc = 0
                            for kb in range(K_BLOCKS):
                                q_seg = Q_codes[h, q_global, kb*16:(kb+1)*16]
                                k_seg = K_codes[h, key, kb*16:(kb+1)*16]
                                dot = 0
                                for el in range(16):
                                    dot += fp4_mult_lut_py(int(q_seg[el]), int(k_seg[el]))
                                res_scale = int(Q_scales[h, q_global, kb]) + int(K_scales[h, key, kb])
                                q_exp = res_scale // 8
                                r_exp = res_scale % 8
                                scaled = csd_scaler_py(dot, r_exp)
                                aligned = shifter_py(scaled, q_exp, frac_bits=16)
                                score_acc = sat_add32(score_acc, int(aligned))
                            score_16.append(int(score_acc))

                        score_nat_16 = [sat32((s * LN2_Q16) >> 16) for s in score_16]
                        row_start = (kj == 0)
                        p_codes, p_blk, l_contrib, resc_req, resc_q, resc_r, m_next = \
                            softmax_requantizer_py(score_nat_16, m_run[m], row_start)

                        if row_start:
                            l_run[m] = l_contrib
                        elif resc_req:
                            l_rescaled = csd_rescaler_32b_py(int(l_run[m]), resc_q, resc_r)
                            l_run[m] = sat_add32(int(l_rescaled), int(l_contrib))
                        else:
                            l_run[m] = sat_add32(int(l_run[m]), int(l_contrib))
                        m_run[m] = m_next

                        P_codes.append(p_codes)
                        P_scales.append(p_blk)
                        P_resc.append((resc_req, resc_q, resc_r))

                    # Mode 10: P * V
                    for m in range(M_Q):
                        p_codes = P_codes[m]
                        p_scale = P_scales[m]
                        resc_req, resc_q, resc_r = P_resc[m]

                        if resc_req:
                            for l in range(16):
                                O_slice[m, l] = csd_rescaler_32b_py(
                                    int(O_slice[m, l]), resc_q, resc_r)

                        for l in range(16):
                            dot = 0
                            for k in range(16):
                                dot += fp4_mult_lut_py(int(p_codes[k]),
                                                       int(V_codes[h, dvi, kj, l, k]))
                            res_scale = int(p_scale) + int(V_scales[h, dvi, kj, l])
                            q_exp = res_scale // 8
                            r_exp = res_scale % 8
                            scaled = csd_scaler_py(dot, r_exp)
                            aligned = shifter_py(scaled, q_exp, frac_bits=16)
                            O_slice[m, l] = sat_add32(int(O_slice[m, l]), int(aligned))

                # Final normalize
                for m in range(M_Q):
                    q_global = q_base + m
                    lt = int(l_run[m])
                    for l in range(16):
                        v_col = dvi * 16 + l
                        norm_val = recip_div_py(int(O_slice[m, l]), lt)
                        code_fp16 = drain_norm_fp16_py(norm_val, frac_bits=16)
                        golden_fp16[h, q_global, v_col] = code_fp16
                        golden_real[h, q_global, v_col] = fp16_hex_to_float(code_fp16)

                O_all[h, qi, dvi, :, :] = O_slice
                l_all[h, qi, dvi, :] = l_run

    # 6. Save Packets and Golden HEX
    with open(f"{HEX}/q_packets.hex", "w") as f:
        for pkt in Q_packets: f.write(f"{pkt:018X}\n")
    with open(f"{HEX}/k_packets.hex", "w") as f:
        for pkt in K_packets: f.write(f"{pkt:018X}\n")
    with open(f"{HEX}/v_packets.hex", "w") as f:
        for pkt in V_packets: f.write(f"{pkt:018X}\n")

    # Order of golden output in hardware:
    # Outer: h -> qi -> dvi -> m (token) -> 16 lanes FP16 (v_col in dvi slice)
    golden_linear = []
    for h in range(H):
        for qi in range(Q_TILES):
            for dvi in range(DV_TILES):
                for m in range(M_Q):
                    q_global = qi * M_Q + m
                    for l in range(16):
                        v_col = dvi * 16 + l
                        golden_linear.append(int(golden_fp16[h, q_global, v_col]))

    with open(f"{HEX}/golden_core_fp16.hex", "w") as f:
        for val in golden_linear:
            f.write(f"{val:04X}\n")

    # 7. TCDM Preload Image
    preload = []
    # Q
    for h in range(H):
        for q in range(N_Q):
            for kb in range(K_BLOCKS):
                idx = h * (N_Q * K_BLOCKS) + q * K_BLOCKS + kb
                preload.append((tcdm_addr(q_word(h, q, kb), Q_BANK),
                                Q_packets[idx],
                                f"Q h={h} q={q} kb={kb}"))
    # K
    for h in range(H):
        for kj in range(KV_TILES):
            for kb in range(K_BLOCKS):
                for lane in range(16):
                    idx = ((h * KV_TILES + kj) * K_BLOCKS + kb) * 16 + lane
                    preload.append((tcdm_addr(k_word(h, kj, kb), lane),
                                    K_packets[idx],
                                    f"K h={h} kj={kj} kb={kb} lane={lane}"))
    # V
    for h in range(H):
        for dvi in range(DV_TILES):
            for kj in range(KV_TILES):
                for lane in range(16):
                    idx = ((h * DV_TILES + dvi) * KV_TILES + kj) * 16 + lane
                    preload.append((tcdm_addr(v_word(h, dvi, kj), lane),
                                    V_packets[idx],
                                    f"V h={h} dvi={dvi} kj={kj} lane={lane}"))

    with open(f"{HEX}/tcdm_preload_addr.hex", "w") as fa, \
         open(f"{HEX}/tcdm_preload_data.hex", "w") as fd:
        for addr, data, _ in preload:
            fa.write(f"{addr:08X}\n")
            fd.write(f"{data:018X}\n")

    # 8. Expected TCDM Output Words (Bank 17)
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
