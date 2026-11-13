#!/usr/bin/env python3
"""
generate_snax_core_flash_tiling_data.py
Bit-exact golden generator for snax_fused_llm_core FULL flash-attention tiling.

Loop nest (matches prove_full_flashattention_tiling_dv.py):
  for h    in 0..H-1            (heads)
  for qi   in 0..Q_TILES-1      (query tiles, M_Q = 8 tokens)
  for dvi  in 0..DV_TILES-1     (d_v slices, 16 channels)
    reset m, l
    for kj in 0..KV_TILES-1     (KV tiles, 16 keys)
      Mode 01: S = Q_tile * K_kj^T -> softmax -> 8 P blocks -> p_fifo
      Mode 10: pop P -> rescale O + P * V[:, dvi*16:(dvi+1)*16]
    drain O slots 0..7 -> final_normalizer -> 16 channels FP16

Dimensions used by the RTL test:
  H = 2, N_Q = 16 (2 query tiles), N_KV = 32 (2 KV tiles),
  D_K = 32 (2 K blocks), D_V = 32 (2 d_v slices), M_Q = 8.
"""
import os
import sys
import csv
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
    FP4_VALS,
    fp4_code_fields,
    write_matrix_csv,
    write_matrix_with_codes,
    write_packet_digital_csv,
)
from generate_fused_attn_8x8 import (
    LN2_Q16,
    sat32,
    csd_rescaler_32b_py,
    softmax_requantizer_py,
    recip_div_py,
    RECIP_LUT,
    mint_of_py,
)


def main():
    np.random.seed(2026)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "sim", "sim_snax_core_flash")
    os.makedirs(f"{out_dir}/hex", exist_ok=True)
    os.makedirs(f"{out_dir}/csv", exist_ok=True)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    H = 2          # heads
    N_Q = 16       # query tokens
    N_KV = 32      # key/value tokens
    D_K = 32       # head dimension for Q/K
    D_V = 32       # head dimension for V

    M_Q = 8        # queries per tile (8:8 slots)
    N_BLOCK = 16   # keys per KV tile
    K_BLOCK = 16   # elements per K block

    Q_TILES = N_Q // M_Q       # 2
    K_BLOCKS = D_K // K_BLOCK  # 2
    DV_TILES = D_V // 16       # 2
    KV_TILES = N_KV // N_BLOCK # 2

    print("=== Generating snax_fused_llm_core FULL TILING data ===")
    print(f"H={H}, N_Q={N_Q} ({Q_TILES} tiles), N_KV={N_KV} ({KV_TILES} tiles), "
          f"D_K={D_K}, D_V={D_V} ({DV_TILES} slices)")

    # ------------------------------------------------------------------
    # Real inputs
    # ------------------------------------------------------------------
    Q_real = np.random.normal(0, 1.0, (H, N_Q, D_K))
    K_real = np.random.normal(0, 1.0, (H, N_KV, D_K))
    V_real = np.random.normal(0, 1.0, (H, N_KV, D_V))

    # Boost the second KV block so the running max increases at kj=1.
    K_real[:, N_BLOCK:2*N_BLOCK, :] *= 1.85

    # ------------------------------------------------------------------
    # Quantize Q: [head][query][K block]
    # ------------------------------------------------------------------
    Q_codes = np.zeros((H, N_Q, D_K), dtype=int)
    Q_fp = np.zeros((H, N_Q, D_K), dtype=float)
    Q_scales = np.zeros((H, N_Q, K_BLOCKS), dtype=int)
    Q_packets = []
    for h in range(H):
        for q in range(N_Q):
            for kb in range(K_BLOCKS):
                codes, vals, s = quantize_block(Q_real[h, q, kb*16:(kb+1)*16])
                Q_codes[h, q, kb*16:(kb+1)*16] = codes
                Q_fp[h, q, kb*16:(kb+1)*16] = vals
                Q_scales[h, q, kb] = s
                Q_packets.append(encode_72b_packet(codes, s))

    # ------------------------------------------------------------------
    # Quantize K: [head][key][K block]
    # Packet order: [head][KV tile][K block][lane]
    # ------------------------------------------------------------------
    K_codes = np.zeros((H, N_KV, D_K), dtype=int)
    K_fp = np.zeros((H, N_KV, D_K), dtype=float)
    K_scales = np.zeros((H, N_KV, K_BLOCKS), dtype=int)
    for h in range(H):
        for k in range(N_KV):
            for kb in range(K_BLOCKS):
                codes, vals, s = quantize_block(K_real[h, k, kb*16:(kb+1)*16])
                K_codes[h, k, kb*16:(kb+1)*16] = codes
                K_fp[h, k, kb*16:(kb+1)*16] = vals
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

    # ------------------------------------------------------------------
    # Quantize V for the hardware layout:
    # packet = 16 keys of one output channel in the current d_v slice.
    # Order: [head][dvi][KV tile][channel in slice][16 keys]
    # ------------------------------------------------------------------
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

    # Hardware V matrix view [head][key][v_col] for a direct P @ V_hw.
    # This is the same quantization the RTL uses (one scale per 16-key channel block).
    V_hw_codes = np.zeros((H, N_KV, D_V), dtype=int)
    V_hw_real = np.zeros((H, N_KV, D_V), dtype=float)
    for h in range(H):
        for dvi in range(DV_TILES):
            for kj in range(KV_TILES):
                for l in range(16):
                    sc = int(V_scales[h, dvi, kj, l])
                    scale = 2.0 ** (sc / 8.0)
                    for k in range(16):
                        key = kj * 16 + k
                        v_col = dvi * 16 + l
                        code = int(V_codes[h, dvi, kj, l, k])
                        _, _, _, nominal = fp4_code_fields(code)
                        V_hw_codes[h, key, v_col] = code
                        V_hw_real[h, key, v_col] = nominal * scale

    # ------------------------------------------------------------------
    # Input packet HEX
    # ------------------------------------------------------------------
    with open(f"{out_dir}/hex/q_packets.hex", "w") as f:
        for pkt in Q_packets:
            f.write(f"{pkt:018X}\n")
    with open(f"{out_dir}/hex/k_packets.hex", "w") as f:
        for pkt in K_packets:
            f.write(f"{pkt:018X}\n")
    with open(f"{out_dir}/hex/v_packets.hex", "w") as f:
        for pkt in V_packets:
            f.write(f"{pkt:018X}\n")

    # ------------------------------------------------------------------
    # Golden state
    # ------------------------------------------------------------------
    golden_fp16 = np.zeros((H, N_Q, D_V), dtype=np.uint16)
    golden_real = np.zeros((H, N_Q, D_V), dtype=float)

    score_raw = np.zeros((H, N_Q, KV_TILES, 16), dtype=np.int64)
    score_ln2 = np.zeros((H, N_Q, KV_TILES, 16), dtype=np.int64)

    p_codes_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q, 16), dtype=int)
    p_blk_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=int)
    p_csum_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=int)
    p_lc_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=np.int64)
    p_resc_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=int)
    p_rq_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=int)
    p_rr_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=int)
    p_m_all = np.zeros((H, Q_TILES, DV_TILES, KV_TILES, M_Q), dtype=np.int64)

    O_all = np.zeros((H, Q_TILES, DV_TILES, M_Q, 16), dtype=np.int32)
    l_all = np.zeros((H, Q_TILES, DV_TILES, M_Q), dtype=np.int32)

    # ------------------------------------------------------------------
    # Hardware loop nest
    # ------------------------------------------------------------------
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

                    # ------------------------- Mode 01: Q * K^T -> softmax
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
                        score_raw[h, q_global, kj, :] = np.array(score_16, dtype=np.int64)
                        score_ln2[h, q_global, kj, :] = np.array(score_nat_16, dtype=np.int64)

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

                        p_codes_all[h, qi, dvi, kj, m, :] = p_codes
                        p_blk_all[h, qi, dvi, kj, m] = p_blk
                        p_csum_all[h, qi, dvi, kj, m] = sum(mint_of_py(int(c)) for c in p_codes)
                        p_lc_all[h, qi, dvi, kj, m] = l_contrib
                        p_resc_all[h, qi, dvi, kj, m] = 1 if resc_req else 0
                        p_rq_all[h, qi, dvi, kj, m] = resc_q
                        p_rr_all[h, qi, dvi, kj, m] = resc_r
                        p_m_all[h, qi, dvi, kj, m] = m_next

                    # ------------------------- Mode 10: rescale O + P * V slice
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

                # ------------------------- final normalize this d_v slice
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

    # ------------------------------------------------------------------
    # Golden HEX: order [head][query][v_col], matching the TB index
    # ------------------------------------------------------------------
    with open(f"{out_dir}/hex/golden_core_fp16.hex", "w") as f:
        for h in range(H):
            for q in range(N_Q):
                for v_col in range(D_V):
                    f.write(f"{golden_fp16[h, q, v_col]:04X}\n")

    # ------------------------------------------------------------------
    # CSV: golden outputs
    # ------------------------------------------------------------------
    csv_dir = f"{out_dir}/csv"

    with open(f"{csv_dir}/golden_attn.csv", "w") as f:
        f.write("head,query_tile,dv_slice,query_in_tile,query_global,lane,v_col_global,"
                "accum_unnorm,l_total,norm_q16,golden_fp16_hex,golden_real\n")
        for h in range(H):
            for qi in range(Q_TILES):
                for dvi in range(DV_TILES):
                    for m in range(M_Q):
                        q_global = qi * M_Q + m
                        lt = int(l_all[h, qi, dvi, m])
                        for l in range(16):
                            v_col = dvi * 16 + l
                            o = int(O_all[h, qi, dvi, m, l])
                            norm = recip_div_py(o, lt)
                            f.write(f"{h},{qi},{dvi},{m},{q_global},{l},{v_col},{o},{lt},{norm},"
                                    f"0x{golden_fp16[h, q_global, v_col]:04X},"
                                    f"{golden_real[h, q_global, v_col]:.7f}\n")

    with open(f"{csv_dir}/golden_output_combined.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "query_tile", "dv_slice", "query_in_tile", "query_global",
                    "lane", "v_col_global", "o_q16", "o_float", "l_q16", "l_float",
                    "norm_q16", "norm_float", "inv_l_mant", "inv_l_exp",
                    "fp16_hex", "fp16_real"])
        for h in range(H):
            for qi in range(Q_TILES):
                for dvi in range(DV_TILES):
                    for m in range(M_Q):
                        q_global = qi * M_Q + m
                        lt = int(l_all[h, qi, dvi, m])
                        lzc = 0
                        for b in range(30, -1, -1):
                            if (lt >> b) & 1:
                                break
                            lzc += 1
                        l_norm = (lt << lzc) & 0x7FFFFFFF
                        inv = RECIP_LUT[(l_norm >> 25) & 0x1F] if lt > 0 else 0
                        for l in range(16):
                            v_col = dvi * 16 + l
                            o = int(O_all[h, qi, dvi, m, l])
                            norm = recip_div_py(o, lt)
                            w.writerow([h, qi, dvi, m, q_global, l, v_col,
                                        o, f"{o / 65536.0:.9f}",
                                        lt, f"{lt / 65536.0:.9f}",
                                        norm, f"{norm / 65536.0:.9f}",
                                        inv, lzc,
                                        f"0x{int(golden_fp16[h, q_global, v_col]):04X}",
                                        f"{golden_real[h, q_global, v_col]:.9f}"])

    with open(f"{csv_dir}/golden_snax_core_flash.csv", "w") as f:
        f.write("head,query_tile,dv_slice,query_in_tile,query_global,ch_in_slice,"
                "v_col_global,golden_fp16_hex,golden_real\n")
        for h in range(H):
            for qi in range(Q_TILES):
                for dvi in range(DV_TILES):
                    for m in range(M_Q):
                        q_global = qi * M_Q + m
                        for l in range(16):
                            v_col = dvi * 16 + l
                            f.write(f"{h},{qi},{dvi},{m},{q_global},{l},{v_col},"
                                    f"0x{golden_fp16[h, q_global, v_col]:04X},"
                                    f"{golden_real[h, q_global, v_col]:.7f}\n")

    # ------------------------------------------------------------------
    # CSV: Q/K/V matrices (one file per head, like the 8x8 matrix CSVs)
    # ------------------------------------------------------------------
    for h in range(H):
        write_matrix_csv(f"{csv_dir}/q_matrix_fp_h{h}.csv", "Query", "Q",
                         [f"E{i}" for i in range(D_K)], Q_real[h])
        write_matrix_csv(f"{csv_dir}/q_matrix_quantized_h{h}.csv", "Query", "Q",
                         [f"E{i}" for i in range(D_K)], Q_fp[h])
        write_matrix_with_codes(f"{csv_dir}/q_matrix_fp_digital_h{h}.csv", "Query", "Q",
                                [f"E{i}" for i in range(D_K)], Q_fp[h], Q_codes[h])

        write_matrix_csv(f"{csv_dir}/k_matrix_fp_h{h}.csv", "Key", "K",
                         [f"E{i}" for i in range(D_K)], K_real[h])
        write_matrix_csv(f"{csv_dir}/k_matrix_quantized_h{h}.csv", "Key", "K",
                         [f"E{i}" for i in range(D_K)], K_fp[h])
        write_matrix_with_codes(f"{csv_dir}/k_matrix_fp_digital_h{h}.csv", "Key", "K",
                                [f"E{i}" for i in range(D_K)], K_fp[h], K_codes[h])

        write_matrix_csv(f"{csv_dir}/v_matrix_fp_h{h}.csv", "Key", "V",
                         [f"Dim_{i}" for i in range(D_V)], V_real[h])
        write_matrix_csv(f"{csv_dir}/v_matrix_hw_quantized_h{h}.csv", "Key", "V",
                         [f"Dim_{i}" for i in range(D_V)], V_hw_real[h])
        write_matrix_with_codes(f"{csv_dir}/v_matrix_hw_fp_digital_h{h}.csv", "Key", "V",
                                [f"Dim_{i}" for i in range(D_V)], V_hw_real[h], V_hw_codes[h])

    # ------------------------------------------------------------------
    # CSV: Q/K packet digital (per head)
    # ------------------------------------------------------------------
    for h in range(H):
        write_packet_digital_csv(f"{csv_dir}/q_packets_digital_h{h}.csv", "q",
                                 Q_codes[h], K_BLOCKS, N_Q, "lane", "Q", Q_scales[h])
        write_packet_digital_csv(f"{csv_dir}/k_packets_digital_h{h}.csv", "k",
                                 K_codes[h], K_BLOCKS, N_KV, "lane", "K", K_scales[h])

    # ------------------------------------------------------------------
    # CSV: V hardware packets and block quantized view
    # ------------------------------------------------------------------
    with open(f"{csv_dir}/v_packets_digital.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["head", "dv_slice", "block", "dim", "v_col_global", "pkt_idx",
                  "scale_hex", "scale_value", "data_hex", "packet_hex"]
        for i in range(16):
            header += [f"K{i}_code_hex", f"K{i}_real"]
        w.writerow(header)
        pkt_idx = 0
        for h in range(H):
            for dvi in range(DV_TILES):
                for kj in range(KV_TILES):
                    for l in range(16):
                        nibbles = [int(x) & 0xF for x in V_codes[h, dvi, kj, l, :]]
                        sc = int(V_scales[h, dvi, kj, l])
                        data64 = 0
                        for i, nib in enumerate(nibbles):
                            data64 |= (nib & 0xF) << (i * 4)
                        packet = (sc & 0xFF) << 64 | data64
                        scale = 2.0 ** (sc / 8.0)
                        dim = dvi * 16 + l
                        row = [h, dvi, kj, dim, dim, pkt_idx,
                               f"0x{sc & 0xFF:02X}", f"{scale:.9f}",
                               f"0x{data64:016X}", f"0x{packet:018X}"]
                        for nib in nibbles:
                            _, _, _, nominal = fp4_code_fields(nib)
                            row += [f"0x{nib:X}", f"{nominal * scale:.9f}"]
                        w.writerow(row)
                        pkt_idx += 1

    with open(f"{csv_dir}/v_block_quantized.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "dv_slice", "block", "dim", "v_col_global",
                    "key", "code_hex", "real", "scale"])
        for h in range(H):
            for dvi in range(DV_TILES):
                for kj in range(KV_TILES):
                    for l in range(16):
                        sc = int(V_scales[h, dvi, kj, l])
                        scale = 2.0 ** (sc / 8.0)
                        dim = dvi * 16 + l
                        for k in range(16):
                            code = int(V_codes[h, dvi, kj, l, k])
                            _, _, _, nominal = fp4_code_fields(code)
                            w.writerow([h, dvi, kj, dim, dim, k, f"0x{code:X}",
                                        f"{nominal * scale:.9f}", f"{scale:.9f}"])

    # ------------------------------------------------------------------
    # CSV: QK scores, P blocks, l state
    # ------------------------------------------------------------------
    with open(f"{csv_dir}/qk_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "query_global", "kv_block", "key",
                    "raw_q16", "raw_float", "ln2_q16", "ln2_float"])
        for h in range(H):
            for q in range(N_Q):
                for kj in range(KV_TILES):
                    for l in range(16):
                        raw = int(score_raw[h, q, kj, l])
                        nat = int(score_ln2[h, q, kj, l])
                        w.writerow([h, q, kj, l, raw, f"{raw / 65536.0:.9f}",
                                    nat, f"{nat / 65536.0:.9f}"])

    with open(f"{csv_dir}/p_blocks.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["head", "query_tile", "dv_slice", "query_in_tile", "query_global",
                  "kv_block", "p_blk", "csum", "l_contrib",
                  "resc_valid", "resc_q", "resc_r", "m_new"]
        header += [f"K{i}_code" for i in range(16)]
        header += [f"K{i}_level" for i in range(16)]
        header += [f"K{i}_p" for i in range(16)]
        w.writerow(header)
        for h in range(H):
            for qi in range(Q_TILES):
                for dvi in range(DV_TILES):
                    for kj in range(KV_TILES):
                        for m in range(M_Q):
                            q_global = qi * M_Q + m
                            codes = p_codes_all[h, qi, dvi, kj, m]
                            levels = [FP4_VALS[int(c)] for c in codes]
                            sp = 2.0 ** (float(p_blk_all[h, qi, dvi, kj, m]) / 8.0)
                            pvals = [float(lv) * sp for lv in levels]
                            row = [h, qi, dvi, m, q_global, kj,
                                   int(p_blk_all[h, qi, dvi, kj, m]),
                                   int(p_csum_all[h, qi, dvi, kj, m]),
                                   int(p_lc_all[h, qi, dvi, kj, m]),
                                   int(p_resc_all[h, qi, dvi, kj, m]),
                                   int(p_rq_all[h, qi, dvi, kj, m]),
                                   int(p_rr_all[h, qi, dvi, kj, m]),
                                   int(p_m_all[h, qi, dvi, kj, m])]
                            row += [int(c) for c in codes]
                            row += [f"{lv:.9f}" for lv in levels]
                            row += [f"{pv:.9f}" for pv in pvals]
                            w.writerow(row)

    with open(f"{csv_dir}/l_state.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "query_tile", "dv_slice", "query_in_tile",
                    "query_global", "l_q16", "l_float"])
        for h in range(H):
            for qi in range(Q_TILES):
                for dvi in range(DV_TILES):
                    for m in range(M_Q):
                        q_global = qi * M_Q + m
                        lt = int(l_all[h, qi, dvi, m])
                        w.writerow([h, qi, dvi, m, q_global, lt, f"{lt / 65536.0:.9f}"])

    print(f"[SUCCESS] All HEX and CSV files generated in {out_dir}/")
    print(f"Generated {H} heads x {N_Q} queries x {D_V} channels = "
          f"{H * N_Q * D_V} golden FP16 values.")


if __name__ == "__main__":
    main()
