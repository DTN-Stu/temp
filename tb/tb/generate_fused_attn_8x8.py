#!/usr/bin/env python3
"""
generate_fused_attn_8x8_data.py
Bit-exact Golden Generator for SNAX Fused LLM Core FlashAttention (8:8 Partitioning).

Hardware Setup:
- Core: snax_fused_llm_core (16 lanes, ACC_DEPTH=16)
- Resource Allocation: 8:8 Split
  * Slots 0..7  : Persistent Output Accumulator O (8 Query Tokens)
  * Slots 8..15 : Scratchpad Score Buffer S (8 Query Tokens x 16 Keys)
- Dimensions:
  * M_query = 8 Tokens (Q0..Q7)
  * d_k = 32 (2 blocks x 16 elements: kb=0, kb=1)
  * d_v = 16 (1 block x 16 elements: 16 channels, matching 16 lanes)
  * N_seq = 32 Keys/Values (2 blocks: j=0 with 16 keys, j=1 with 16 keys)
"""

import os
import sys
import csv
import math
import struct
import numpy as np

# Ensure local directory is in path to import from generate_prefill_gemm_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_prefill_gemm_data import (
    FP4_VALS,
    FP4_MID,
    encode_72b_packet,
    float_to_fp16_hex,
    fp16_hex_to_float,
    make_gaussian_matrix,
    quantize_block,
    fp4_mult_lut_py,
    csd_scaler_py,
    shifter_py,
    sat_add32,
    drain_norm_fp16_py,
    fp4_code_fields,
    write_matrix_csv,
    write_matrix_with_codes,
    write_packet_digital_csv,
)

# Constants matching RTL
LN2_Q16 = 45426
CSD_Q5 = [32, 35, 38, 41, 46, 49, 54, 59]

TH_6P0 = 17238
TH_4P0 = 50961
TH_3P0 = 82774
TH_2P0 = 116497
TH_1P5 = 148310
TH_1P0 = 196608
TH_0P5 = 300480

P_OFFSET = -21
P_RND = 4096  # 1 << 12

RECIP_LUT = [
    32768, 31775, 30840, 29960, 29127, 28339, 27592, 26884,
    26214, 25579, 24976, 24403, 23859, 23341, 22847, 22376,
    21926, 21495, 21083, 20687, 20308, 19944, 19594, 19258,
    18934, 18622, 18322, 18032, 17752, 17482, 17220, 16968
]

def sat32(v):
    v = int(v)
    if v > 2147483647:
        return 2147483647
    elif v < -2147483648:
        return -2147483648
    return v

def csd_rescaler_32b_py(val_in, resc_q, resc_r):
    """Bit-exact mirror of vsq_rescaler_32b.sv (38-bit CSA, no saturation)."""
    m38 = (1 << 38) - 1

    def s38(x):
        x &= m38
        return x - (1 << 38) if x & (1 << 37) else x

    def s32(x):
        x &= 0xFFFFFFFF
        return x - (1 << 32) if x & (1 << 31) else x

    val_in  = int(val_in)
    resc_q  = int(resc_q)
    resc_r  = int(resc_r)
    ve_s    = s38(val_in)

    if (resc_r & 0x6) == 0x6:
        t0 = (ve_s << 6) & m38
    else:
        t0 = (ve_s << 5) & m38

    if resc_r == 0:
        t1m, sub1, t2m, sub2 = 0, 0, 0, 0
    elif resc_r == 1:
        t1m, sub1, t2m, sub2 = (ve_s << 1) & m38, 0, ve_s & m38, 0
    elif resc_r == 2:
        t1m, sub1, t2m, sub2 = (ve_s << 2) & m38, 0, (ve_s << 1) & m38, 0
    elif resc_r == 3:
        t1m, sub1, t2m, sub2 = (ve_s << 3) & m38, 0, ve_s & m38, 0
    elif resc_r == 4:
        t1m, sub1, t2m, sub2 = (ve_s << 4) & m38, 0, (ve_s << 1) & m38, 1
    elif resc_r == 5:
        t1m, sub1, t2m, sub2 = (ve_s << 4) & m38, 0, ve_s & m38, 0
    elif resc_r == 6:
        t1m, sub1, t2m, sub2 = (ve_s << 3) & m38, 1, (ve_s << 1) & m38, 1
    else:
        t1m, sub1, t2m, sub2 = (ve_s << 2) & m38, 1, ve_s & m38, 1

    t1 = ((~t1m) & m38) if sub1 else t1m
    t2 = ((~t2m) & m38) if sub2 else t2m

    csa_sum  = (t0 ^ t1 ^ t2) & m38
    csa_cout = ((t0 & t1) | (t1 & t2) | (t0 & t2)) & m38
    carry    = ((csa_cout << 1) | sub1) & m38
    prod     = (csa_sum + carry + sub2) & m38

    net = 32 if resc_q < -26 else 5 + ((-resc_q) & 0x1F)
    rounded = s38(prod) + (1 << (net - 1))
    shifted = s38(rounded) >> net
    return s32(shifted)

def mag_of_py(d):
    if d <= TH_6P0: return 7
    elif d <= TH_4P0: return 6
    elif d <= TH_3P0: return 5
    elif d <= TH_2P0: return 4
    elif d <= TH_1P5: return 3
    elif d <= TH_1P0: return 2
    elif d <= TH_0P5: return 1
    else: return 0

def mint_of_py(mag):
    table = [0, 1, 2, 3, 4, 6, 8, 12]
    return table[mag]

def softmax_requantizer_py(score_in_16, m_prev, row_start):
    """Bit-exact model of softmax_requantizer.sv for 1 block of 16 keys"""
    score_in_16 = [int(s) for s in score_in_16]
    m_prev = int(m_prev)
    blk_max = max(score_in_16)

    if row_start:
        m_next = blk_max
        resc_req = False
    else:
        m_next = max(blk_max, m_prev)
        resc_req = (blk_max > m_prev)

    resc_delta = int(m_prev) - int(m_next)
    resc_oct = resc_delta >> 16
    if resc_oct < -128: resc_q = -128
    elif resc_oct > 127: resc_q = 127
    else: resc_q = resc_oct

    resc_r = (resc_delta >> 13) & 0x7

    # Block scale
    blk_max_minus_m = blk_max - m_next
    p_wide = ((blk_max_minus_m + P_RND) >> 13) + P_OFFSET
    if p_wide < -128: p_blk = -128
    elif p_wide > 127: p_blk = 127
    else: p_blk = int(p_wide)

    # Magnitude codes & csum
    codes = []
    csum = 0
    for t in range(16):
        d_t = blk_max - score_in_16[t]
        mag = mag_of_py(d_t)
        codes.append(mag)
        csum += mint_of_py(mag)

    # l_contrib
    q_s = p_blk >> 3
    r_s = p_blk & 7
    prod = csum * CSD_Q5[r_s]
    sh = q_s + 16
    if sh > 15: sh = 15
    if sh >= 6:
        l_contrib = prod << (sh - 6)
    else:
        l_contrib = prod >> (6 - sh)

    return codes, p_blk, l_contrib, resc_req, resc_q, resc_r, m_next

def recip_div_py(accum_val, l_total):
    """Bit-exact model of reciprocal generator + final_normalizer multiplier"""
    accum_val = int(accum_val)
    l_total = int(l_total)
    if l_total <= 0:
        return 0

    lzc = 0
    for b in range(30, -1, -1):
        if (l_total >> b) & 1:
            break
        lzc += 1

    l_norm = (l_total << lzc) & 0x7FFFFFFF
    lut_idx = (l_norm >> 25) & 0x1F
    inv_l_mant = RECIP_LUT[lut_idx]
    inv_l_exp = lzc

    raw_mult = accum_val * inv_l_mant

    if inv_l_exp <= 29:
        shifted = raw_mult >> (29 - inv_l_exp)          # no rounding (RTL)
    else:
        shifted = raw_mult << (inv_l_exp - 29)

    return sat32(shifted)

def main():
    np.random.seed(1234)

    out_dir = "../sim/sim_fused_attn_8x8"
    os.makedirs(f"{out_dir}/hex", exist_ok=True)
    os.makedirs(f"{out_dir}/csv", exist_ok=True)

    print("=== Generating Data for Fused FlashAttention 8:8 Verification ===")
    print("M_query=8, d_k=32, d_v=16, N_seq=32 (2 blocks of 16 keys/values)")

    # 1. Generate Q (8 x 32), K (32 x 32), V (32 x 16)
    Q_real = np.random.normal(0, 1.0, (8, 32))
    K_real = np.random.normal(0, 1.0, (32, 32))
    V_real = np.random.normal(0, 1.0, (32, 16))

    # Boost block 1 scores to guarantee online rescaling triggers (m_new > m_old)
    K_real[16:32, :] *= 1.8

    # Quantize Q, K, V
    Q_packets = []
    Q_codes = np.zeros((8, 32), dtype=int)
    Q_scales = np.zeros((8, 2), dtype=int)
    Q_fp = np.zeros((8, 32), dtype=float)
    for m in range(8):
        for kb in range(2):
            seg = Q_real[m, kb*16:(kb+1)*16]
            codes, vals, s = quantize_block(seg)
            Q_codes[m, kb*16:(kb+1)*16] = codes
            Q_scales[m, kb] = s
            Q_fp[m, kb*16:(kb+1)*16] = vals
            Q_packets.append(encode_72b_packet(codes, s))

    K_codes = np.zeros((32, 32), dtype=int)
    K_scales = np.zeros((32, 2), dtype=int)
    K_fp = np.zeros((32, 32), dtype=float)
    for k_idx in range(32):
        for kb in range(2):
            seg = K_real[k_idx, kb*16:(kb+1)*16]
            codes, vals, s = quantize_block(seg)
            K_codes[k_idx, kb*16:(kb+1)*16] = codes
            K_scales[k_idx, kb] = s
            K_fp[k_idx, kb*16:(kb+1)*16] = vals

    # K packets must be block-major: [key block][K sub-block][key lane].
    K_packets = []
    for j in range(2):
        for kb in range(2):
            for lane in range(16):
                k_idx = j * 16 + lane
                K_packets.append(encode_72b_packet(
                    K_codes[k_idx, kb*16:(kb+1)*16],
                    int(K_scales[k_idx, kb])))

    # V packet for P*V must contain the 16 KEYS of one output dimension,
    # with one scale per 16-key block: [key block][output dim][key].
    V_packets = []
    V_codes = np.zeros((2, 16, 16), dtype=int)
    V_scales = np.zeros((2, 16), dtype=int)
    V_fp_block = np.zeros((2, 16, 16), dtype=float)
    for j in range(2):
        for l in range(16):
            seg = V_real[j*16:(j+1)*16, l]
            codes, vals, s = quantize_block(seg)
            V_codes[j, l, :] = codes
            V_scales[j, l] = s
            V_fp_block[j, l, :] = vals
            V_packets.append(encode_72b_packet(codes, s))

    # Original V matrix view (one scale per key row), for CSV debug only.
    V_codes_key = np.zeros((32, 16), dtype=int)
    V_fp_key = np.zeros((32, 16), dtype=float)
    V_scales_key = np.zeros((32,), dtype=int)
    for v_idx in range(32):
        codes, vals, s = quantize_block(V_real[v_idx, :])
        V_codes_key[v_idx, :] = codes
        V_fp_key[v_idx, :] = vals
        V_scales_key[v_idx] = s

    # Write input HEX files
    with open(f"{out_dir}/hex/q_packets.hex", "w") as f:
        for pkt in Q_packets:
            f.write(f"{pkt:018X}\n")

    with open(f"{out_dir}/hex/k_packets.hex", "w") as f:
        for pkt in K_packets:
            f.write(f"{pkt:018X}\n")

    with open(f"{out_dir}/hex/v_packets.hex", "w") as f:
        for pkt in V_packets:
            f.write(f"{pkt:018X}\n")

    # 2. Bit-exact Hardware Simulation
    O_accum = np.zeros((8, 16), dtype=np.int32)
    l_running = np.zeros(8, dtype=np.int32)
    m_running = np.full(8, -2147483648, dtype=np.int32)

    # Debug arrays (one entry per query, KV block, key).
    score_raw   = np.zeros((8, 2, 16), dtype=np.int64)
    score_ln2   = np.zeros((8, 2, 16), dtype=np.int64)
    p_codes_arr = np.zeros((8, 2, 16), dtype=int)
    p_blk_arr   = np.zeros((8, 2), dtype=int)
    p_lc_arr    = np.zeros((8, 2), dtype=np.int64)
    p_csum_arr  = np.zeros((8, 2), dtype=int)
    p_resc_arr  = np.zeros((8, 2), dtype=int)
    p_rq_arr    = np.zeros((8, 2), dtype=int)
    p_rr_arr    = np.zeros((8, 2), dtype=int)
    p_m_arr     = np.zeros((8, 2), dtype=np.int64)

    for j in range(2):
        key_start = j * 16
        key_end = key_start + 16

        # --- MODE 01: S = Q * K^T, then Softmax ---
        P_block_codes = []
        P_block_scales = []
        rescale_sidebands = []

        for m in range(8):
            score_16 = []
            for lane in range(16):
                key_token = key_start + lane
                score = 0
                for kb in range(2):
                    dot = 0
                    q_seg = Q_codes[m, kb*16:(kb+1)*16]
                    k_seg = K_codes[key_token, kb*16:(kb+1)*16]
                    for el in range(16):
                        dot += fp4_mult_lut_py(q_seg[el], k_seg[el])
                    res_scale = int(Q_scales[m, kb]) + int(K_scales[key_token, kb])
                    q_exp = res_scale // 8
                    r_exp = res_scale % 8
                    scaled = csd_scaler_py(dot, r_exp)
                    aligned = shifter_py(scaled, q_exp, frac_bits=16)
                    score = sat_add32(score, aligned)
                score_16.append(score)

            # Convert to natural base
            score_nat_16 = [sat32((s * LN2_Q16) >> 16) for s in score_16]

            score_raw[m, j, :] = score_16
            score_ln2[m, j, :] = score_nat_16

            row_start = (j == 0)
            p_codes, p_blk, l_contrib, resc_req, resc_q, resc_r, m_next = softmax_requantizer_py(
                score_nat_16, m_running[m], row_start
            )

            p_codes_arr[m, j, :] = p_codes
            p_blk_arr[m, j]  = p_blk
            p_lc_arr[m, j]   = l_contrib
            p_csum_arr[m, j] = sum(mint_of_py(c) for c in p_codes)
            p_resc_arr[m, j] = 1 if resc_req else 0
            p_rq_arr[m, j]   = resc_q
            p_rr_arr[m, j]   = resc_r
            p_m_arr[m, j]    = m_next

            # Update l
            if row_start:
                l_running[m] = l_contrib
            elif resc_req:
                l_rescaled = csd_rescaler_32b_py(l_running[m], resc_q, resc_r)
                l_running[m] = sat_add32(l_rescaled, l_contrib)
            else:
                l_running[m] = sat_add32(l_running[m], l_contrib)

            m_running[m] = m_next

            if m == 0:
                print(f"GOLD_QK j={j} m={m} raw={score_16}")
                print(f"GOLD_P  j={j} m={m} pblk={p_blk} lc={l_contrib} codes={p_codes}")

            P_block_codes.append(p_codes)
            P_block_scales.append(p_blk)
            rescale_sidebands.append((resc_req, resc_q, resc_r))

        # --- MODE 10: O = Rescale(O) + P * V ---
        for m in range(8):
            p_codes = P_block_codes[m]
            p_scale = P_block_scales[m]
            resc_req, resc_q, resc_r = rescale_sidebands[m]

            # If rescale requested, rescale old O accumulator
            if resc_req:
                for l in range(16):
                    O_accum[m, l] = csd_rescaler_32b_py(O_accum[m, l], resc_q, resc_r)

            # Compute P * V
            for l in range(16):
                v_slice = V_codes[j, l, :]
                v_scale = int(V_scales[j, l])

                dot = 0
                for k in range(16):
                    dot += fp4_mult_lut_py(p_codes[k], v_slice[k])

                res_scale = int(p_scale) + v_scale
                q_exp = res_scale // 8
                r_exp = res_scale % 8
                scaled = csd_scaler_py(dot, r_exp)
                aligned = shifter_py(scaled, q_exp, frac_bits=16)

                O_accum[m, l] = sat_add32(O_accum[m, l], aligned)

    # 3. Final Normalization: O_norm = O_accum / l_running
    golden_fp16_hex = np.zeros((8, 16), dtype=np.uint16)
    golden_real = np.zeros((8, 16), dtype=float)

    for m in range(8):
        l_tot = l_running[m]
        for l in range(16):
            norm_val = recip_div_py(O_accum[m, l], l_tot)
            h = drain_norm_fp16_py(norm_val, frac_bits=16)
            golden_fp16_hex[m, l] = h
            golden_real[m, l] = fp16_hex_to_float(h)

    # Write Golden Output HEX & CSV
    with open(f"{out_dir}/hex/golden_attn_fp16.hex", "w") as f:
        for m in range(8):
            for l in range(16):
                f.write(f"{golden_fp16_hex[m, l]:04X}\n")

    csv_dir = f"{out_dir}/csv"

    with open(f"{csv_dir}/golden_attn.csv", "w") as f:
        f.write("query_idx,lane,accum_unnorm,l_total,norm_q16,golden_fp16_hex,golden_real\n")
        for m in range(8):
            for l in range(16):
                norm = recip_div_py(O_accum[m, l], l_running[m])
                f.write(f"{m},{l},{O_accum[m, l]},{l_running[m]},{norm},"
                        f"0x{golden_fp16_hex[m, l]:04X},{golden_real[m, l]:.7f}\n")

    # ------------------------------------------------------------------
    # Debug CSV: Q/K/V Gaussian, quantized, 4-bit codes, 72-bit packets
    # ------------------------------------------------------------------
    write_matrix_csv(f"{csv_dir}/q_matrix_fp.csv", "Query", "Q",
                     [f"E{i}" for i in range(32)], Q_real)
    write_matrix_csv(f"{csv_dir}/q_matrix_quantized.csv", "Query", "Q",
                     [f"E{i}" for i in range(32)], Q_fp)
    write_matrix_with_codes(f"{csv_dir}/q_matrix_fp_digital.csv", "Query", "Q",
                            [f"E{i}" for i in range(32)], Q_fp, Q_codes)

    write_matrix_csv(f"{csv_dir}/k_matrix_fp.csv", "Key", "K",
                     [f"E{i}" for i in range(32)], K_real)
    write_matrix_csv(f"{csv_dir}/k_matrix_quantized.csv", "Key", "K",
                     [f"E{i}" for i in range(32)], K_fp)
    write_matrix_with_codes(f"{csv_dir}/k_matrix_fp_digital.csv", "Key", "K",
                            [f"E{i}" for i in range(32)], K_fp, K_codes)

    write_matrix_csv(f"{csv_dir}/v_matrix_fp.csv", "Key", "V",
                     [f"Dim_{i}" for i in range(16)], V_real)
    write_matrix_csv(f"{csv_dir}/v_matrix_quantized.csv", "Key", "V",
                     [f"Dim_{i}" for i in range(16)], V_fp_key)
    write_matrix_with_codes(f"{csv_dir}/v_matrix_fp_digital.csv", "Key", "V",
                            [f"Dim_{i}" for i in range(16)], V_fp_key, V_codes_key)

    write_packet_digital_csv(f"{csv_dir}/q_packets_digital.csv", "q",
                             Q_codes, 2, 8, "lane", "Q", Q_scales)
    write_packet_digital_csv(f"{csv_dir}/k_packets_digital.csv", "k",
                             K_codes, 2, 32, "lane", "K", K_scales)

    # Hardware V layout: [key block][output dim][16 keys], one scale per packet.
    with open(f"{csv_dir}/v_packets_digital.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["block", "dim", "pkt_idx", "scale_hex", "scale_value",
                  "data_hex", "packet_hex"]
        for i in range(16):
            header += [f"K{i}_code_hex", f"K{i}_real"]
        w.writerow(header)
        pkt_idx = 0
        for j in range(2):
            for l in range(16):
                nibbles = [int(x) & 0xF for x in V_codes[j, l, :]]
                sc = int(V_scales[j, l])
                data64 = 0
                for i, nib in enumerate(nibbles):
                    data64 |= (nib & 0xF) << (i * 4)
                packet = (sc & 0xFF) << 64 | data64
                scale = 2.0 ** (sc / 8.0)
                row = [j, l, pkt_idx, f"0x{sc & 0xFF:02X}", f"{scale:.9f}",
                       f"0x{data64:016X}", f"0x{packet:018X}"]
                for nib in nibbles:
                    _, _, _, nominal = fp4_code_fields(nib)
                    row += [f"0x{nib:X}", f"{nominal * scale:.9f}"]
                w.writerow(row)
                pkt_idx += 1

    with open(f"{csv_dir}/v_block_quantized.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "dim", "key", "code_hex", "real", "scale"])
        for j in range(2):
            for l in range(16):
                sc = int(V_scales[j, l])
                scale = 2.0 ** (sc / 8.0)
                for k in range(16):
                    code = int(V_codes[j, l, k])
                    _, _, _, nominal = fp4_code_fields(code)
                    w.writerow([j, l, k, f"0x{code:X}",
                                f"{nominal * scale:.9f}", f"{scale:.9f}"])

    # ------------------------------------------------------------------
    # Debug CSV: QK scores, softmax P blocks, l/O
    # ------------------------------------------------------------------
    with open(f"{csv_dir}/qk_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "kv_block", "key", "raw_q16", "raw_float",
                    "ln2_q16", "ln2_float"])
        for m in range(8):
            for j in range(2):
                for l in range(16):
                    raw = int(score_raw[m, j, l])
                    nat = int(score_ln2[m, j, l])
                    w.writerow([m, j, l, raw, f"{raw / 65536.0:.9f}",
                                nat, f"{nat / 65536.0:.9f}"])

    with open(f"{csv_dir}/p_blocks.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["query", "kv_block", "p_blk", "csum", "l_contrib",
                  "resc_valid", "resc_q", "resc_r", "m_new"]
        header += [f"K{i}_code" for i in range(16)]
        header += [f"K{i}_level" for i in range(16)]
        header += [f"K{i}_p" for i in range(16)]
        w.writerow(header)
        for m in range(8):
            for j in range(2):
                codes = p_codes_arr[m, j]
                levels = [FP4_VALS[int(c)] for c in codes]
                sp = 2.0 ** (float(p_blk_arr[m, j]) / 8.0)
                pvals = [float(lv) * sp for lv in levels]
                row = [m, j, int(p_blk_arr[m, j]), int(p_csum_arr[m, j]),
                       int(p_lc_arr[m, j]), int(p_resc_arr[m, j]),
                       int(p_rq_arr[m, j]), int(p_rr_arr[m, j]),
                       int(p_m_arr[m, j])]
                row += [int(c) for c in codes]
                row += [f"{lv:.9f}" for lv in levels]
                row += [f"{pv:.9f}" for pv in pvals]
                w.writerow(row)

    with open(f"{csv_dir}/l_state.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "l_q16", "l_float"])
        for m in range(8):
            w.writerow([m, int(l_running[m]), f"{float(l_running[m]) / 65536.0:.9f}"])

    with open(f"{csv_dir}/golden_output_combined.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "lane", "o_q16", "o_float", "l_q16", "l_float",
                    "norm_q16", "norm_float", "inv_l_mant", "inv_l_exp",
                    "fp16_hex", "fp16_real"])
        for m in range(8):
            lt = int(l_running[m])
            lzc = 0
            for b in range(30, -1, -1):
                if (lt >> b) & 1:
                    break
                lzc += 1
            l_norm = (lt << lzc) & 0x7FFFFFFF
            inv = RECIP_LUT[(l_norm >> 25) & 0x1F]
            for l in range(16):
                o = int(O_accum[m, l])
                norm = recip_div_py(o, lt)
                w.writerow([m, l, o, f"{o / 65536.0:.9f}",
                            lt, f"{lt / 65536.0:.9f}",
                            norm, f"{norm / 65536.0:.9f}",
                            inv, lzc,
                            f"0x{int(golden_fp16_hex[m, l]):04X}",
                            f"{golden_real[m, l]:.9f}"])

    print(f"[SUCCESS] All HEX and CSV files generated in {out_dir}/")
    print(f"Generated 8 Query Tokens x 16 Lanes = 128 Golden FP16 values.")

if __name__ == "__main__":
    main()
