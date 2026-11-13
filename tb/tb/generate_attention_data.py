#!/usr/bin/env python3
"""
generate_attention_data.py

Data + golden generator for one attention query on the SNAX FP4 fused core.

Geometry (one query, 32 keys, head dimension 16):
    Q       : 16 FP4 values (one block-scaled 72-bit packet)
    K       : 32 keys x 16 FP4 values (2 blocks of 16 keys)
    V       : 16 output dims x 32 keys (2 blocks of 16 keys)
    output  : 16 FP4/FP16 values after softmax(Q*K^T) * V

The input values are sampled from N(0, sigma), then block-quantized per 16
elements with the same 1/8-octave Log8 scale used by the GEMM generator.

Three references are produced:
  golden_fp16       : bit-exact Python mirror of the RTL datapath
  ref_fakequant     : exact float attention on the same quantized Q/K/V
  ref_gaussian      : exact float attention on the original Gaussian inputs

The TB compares the RTL output against golden_fp16 (bit-exact) and reports
the RTL-vs-fakequant error metrics.
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_prefill_gemm_data as g

# ---------------------------------------------------------------------------
# Test geometry
# ---------------------------------------------------------------------------
NUM_LANES   = 16
NUM_BLOCKS  = 2
NUM_KEYS    = NUM_BLOCKS * NUM_LANES
BLOCK_SIZE  = 16
SCORE_FRAC  = 16
SIGMA       = 1.0
SEED        = 2026
K_BLOCK1_GAIN = 2.0
KEY_MASKS   = [0xFFFF, 0x00FF]

OUT_DIR = HERE.parent / "sim" / "sim_attn"
HEX_DIR = OUT_DIR / "hex"
CSV_DIR = OUT_DIR / "csv"

# RTL constants
S_MIN     = -0x80000000
P_RND     = 1 << (SCORE_FRAC - 4)
P_OFFSET  = -21
LN2_Q16   = 45426       # ln(2) in Q16.16, same constant as the RTL
THRESHOLDS = [17238, 50961, 82774, 116497, 148310, 196608, 300480]
CSD_Q5     = [32, 35, 38, 41, 46, 49, 54, 59]
RECIP_LUT  = [
    32768, 31775, 30840, 29960, 29127, 28339, 27592, 26884,
    26214, 25579, 24976, 24403, 23859, 23341, 22847, 22376,
    21926, 21495, 21083, 20687, 20308, 19944, 19594, 19258,
    18934, 18622, 18322, 18032, 17752, 17482, 17220, 16968,
]


def signed32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & (1 << 31) else x


def signed38(x):
    x &= (1 << 38) - 1
    return x - (1 << 38) if x & (1 << 37) else x


def mag_of(d):
    if d <= THRESHOLDS[0]: return 7
    if d <= THRESHOLDS[1]: return 6
    if d <= THRESHOLDS[2]: return 5
    if d <= THRESHOLDS[3]: return 4
    if d <= THRESHOLDS[4]: return 3
    if d <= THRESHOLDS[5]: return 2
    if d <= THRESHOLDS[6]: return 1
    return 0


def mint_of(mag):
    return [0, 1, 2, 3, 4, 6, 8, 12][mag]


def dot16(codes_a, codes_b):
    total = 0
    for a, b in zip(codes_a, codes_b):
        total += g.fp4_mult_lut_py(int(a), int(b))
    return total


def matmul_q16(product, scale_a, scale_b):
    """dot * 2^((scale_a+scale_b)/8), Q16.16 (matmul Stage 1)."""
    res = int(scale_a) + int(scale_b)
    q = res // 8
    r = res - 8 * q
    return g.shifter_py(g.csd_scaler_py(int(product), int(r)), q,
                        frac_bits=SCORE_FRAC)


def scale_ln2_q16(score):
    """Mirror of the core's score_for_softmax = sat32(score*LN2 >>> 16)."""
    return (int(score) * LN2_Q16) >> SCORE_FRAC


def rescaler_q16(val, resc_q, resc_r):
    """Bit-exact mirror of vsq_rescaler_32b.sv."""
    m38 = (1 << 38) - 1
    ve = int(val) & m38
    ve_s = signed38(ve)

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

    csa_sum = (t0 ^ t1 ^ t2) & m38
    csa_cout = ((t0 & t1) | (t1 & t2) | (t0 & t2)) & m38
    carry = ((csa_cout << 1) | sub1) & m38
    prod = (csa_sum + carry + sub2) & m38

    if int(resc_q) < -26:
        net = 32
    else:
        net = 5 + ((-int(resc_q)) & 0x1F)

    rounded = signed38(prod) + (1 << (net - 1))
    shifted = signed38(rounded) >> net
    return signed32(shifted)


def softmax_block(scores, key_mask, m_old, row_start):
    """Python mirror of softmax_requantizer.sv for one 16-key block."""
    masked_scores = [int(scores[i]) if (key_mask >> i) & 1 else S_MIN
                     for i in range(BLOCK_SIZE)]
    any_valid = key_mask != 0
    blk_max = max(masked_scores) if any_valid else S_MIN

    if not any_valid:
        m_new = m_old
        resc_valid = False
        resc_q = 0
        resc_r = 0
    elif row_start:
        m_new = blk_max
        resc_valid = False
        resc_q = 0
        resc_r = 0
    else:
        m_new = max(m_old, blk_max)
        resc_valid = blk_max > m_old
        delta = m_old - m_new
        octv = delta >> SCORE_FRAC
        resc_q = max(-128, min(127, octv))
        resc_r = (delta >> 13) & 0x7

    p_wide = ((blk_max - m_new + P_RND) >> 13) + P_OFFSET
    p_blk = max(-128, min(127, p_wide))

    codes = [0] * BLOCK_SIZE
    csum = 0
    for i in range(BLOCK_SIZE):
        dist = blk_max - masked_scores[i]
        mag = mag_of(dist)
        if (key_mask >> i) & 1:
            codes[i] = mag
            csum += mint_of(mag)

    csum &= 0xFF
    r_idx = p_blk & 0x7
    q_s = p_blk >> 3
    prod = csum * CSD_Q5[r_idx]
    sh = q_s + SCORE_FRAC
    if sh > 15:
        sh = 15
    if sh >= 6:
        l_contrib = prod << (sh - 6)
    else:
        l_contrib = prod >> (6 - sh)
    l_contrib = signed32(l_contrib)

    return {
        "m_new": m_new,
        "resc_valid": resc_valid,
        "resc_q": resc_q,
        "resc_r": resc_r,
        "p_blk": p_blk,
        "codes": codes,
        "csum": csum,
        "l_contrib": l_contrib,
        "any_valid": any_valid,
        "blk_max": blk_max,
    }


def final_norm_fp16(o_q16, l_q16):
    """Python mirror of final_normalizer.sv + logscale_lazy_drain_norm."""
    if l_q16 <= 0:
        return 0x0000
    msb = int(l_q16).bit_length() - 1
    lzc = 30 - msb
    l_norm = ((int(l_q16) & 0x7FFFFFFF) << lzc) & 0x7FFFFFFF
    idx = (l_norm >> 25) & 0x1F
    inv = RECIP_LUT[idx]
    raw = int(o_q16) * inv
    if lzc <= 29:
        scaled = raw >> (29 - lzc)
    else:
        scaled = raw << (lzc - 29)
    scaled = max(-(1 << 31), min((1 << 31) - 1, scaled))
    return g.drain_norm_fp16_py(scaled, frac_bits=SCORE_FRAC)


def exact_attention(q, k, v, masks):
    """Exact float attention on given (already dequantized) Q/K/V."""
    scores = np.asarray(q, dtype=float) @ np.asarray(k, dtype=float).T
    keys = list(range(NUM_KEYS))
    for key in keys:
        b = key // BLOCK_SIZE
        i = key % BLOCK_SIZE
        if ((masks[b] >> i) & 1) == 0:
            scores[key] = -np.inf
    s_max = np.max(scores)
    e = np.exp(scores - s_max)
    p = e / np.sum(e)
    o = p @ np.asarray(v, dtype=float).T
    return o


def write_csv_matrix(path, row_header, row_prefix, col_labels, mat, fmt=".6f"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([row_header] + col_labels)
        for r in range(mat.shape[0]):
            w.writerow([f"{row_prefix}_{r}"] +
                       [f"{mat[r, c]:{fmt}}" for c in range(mat.shape[1])])


def main():
    np.random.seed(SEED)
    HEX_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Gaussian inputs, with a 2x scale on K block 1 so the running max
    # increases and the online-rescale path is exercised.
    # ------------------------------------------------------------------
    q_gauss = np.random.normal(0.0, SIGMA, BLOCK_SIZE)
    k_gauss = np.random.normal(0.0, SIGMA, (NUM_KEYS, BLOCK_SIZE))
    k_gauss[NUM_LANES:] *= K_BLOCK1_GAIN
    v_gauss = np.random.normal(0.0, SIGMA, (NUM_LANES, NUM_KEYS))

    q_codes, q_fp, q_scale = g.quantize_block(q_gauss)
    k_codes, k_fp, k_scale = g.quantize_matrix(k_gauss, 1, block_axis=1)
    v_codes, v_fp, v_scale = g.quantize_matrix(v_gauss, NUM_BLOCKS, block_axis=1)

    q_scale = int(q_scale)

    # ------------------------------------------------------------------
    # QK scores, exactly as the matmul engine produces them (Q16.16).
    # ------------------------------------------------------------------
    scores = np.zeros((NUM_BLOCKS, BLOCK_SIZE), dtype=np.int64)
    for b in range(NUM_BLOCKS):
        for i in range(BLOCK_SIZE):
            key = b * BLOCK_SIZE + i
            dot = dot16(q_codes, k_codes[key])
            scores[b, i] = matmul_q16(dot, q_scale, int(k_scale[key, 0]))

    # The RTL converts the base-2 score to base-e before softmax:
    #   score_for_softmax = sat32(score * ln2_Q16 >>> 16)
    scores_ln2 = np.array(
        [[scale_ln2_q16(int(scores[b, i])) for i in range(BLOCK_SIZE)]
         for b in range(NUM_BLOCKS)], dtype=np.int64)

    # ------------------------------------------------------------------
    # Fused exp-free softmax, l accumulation and P blocks.
    # ------------------------------------------------------------------
    m_old = S_MIN
    l_q16 = 0
    p_blocks = []
    for b in range(NUM_BLOCKS):
        blk = softmax_block(scores_ln2[b], KEY_MASKS[b], m_old, b == 0)
        m_old = blk["m_new"]
        p_blocks.append(blk)

        if b == 0:
            l_q16 = blk["l_contrib"]
        elif blk["resc_valid"]:
            l_q16 = g.sat_add32(
                rescaler_q16(l_q16, blk["resc_q"], blk["resc_r"]),
                blk["l_contrib"])
        else:
            l_q16 = g.sat_add32(l_q16, blk["l_contrib"])
        l_q16 = signed32(l_q16)

    # ------------------------------------------------------------------
    # P*V, with online O rescale before each block.
    # ------------------------------------------------------------------
    o_q16 = np.zeros(NUM_LANES, dtype=np.int64)
    for b in range(NUM_BLOCKS):
        blk = p_blocks[b]
        if blk["resc_valid"]:
            for d in range(NUM_LANES):
                o_q16[d] = rescaler_q16(int(o_q16[d]),
                                        blk["resc_q"], blk["resc_r"])
        for d in range(NUM_LANES):
            v_slice = v_codes[d, b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
            dot = dot16(blk["codes"], v_slice)
            prod = matmul_q16(dot, blk["p_blk"], int(v_scale[d, b]))
            o_q16[d] = g.sat_add32(int(o_q16[d]), prod)
            o_q16[d] = signed32(int(o_q16[d]))

    golden_fp16 = np.zeros(NUM_LANES, dtype=np.uint16)
    golden_real = np.zeros(NUM_LANES, dtype=float)
    for d in range(NUM_LANES):
        h = final_norm_fp16(int(o_q16[d]), int(l_q16))
        golden_fp16[d] = h
        golden_real[d] = g.fp16_hex_to_float(h)

    # ------------------------------------------------------------------
    # Float references.
    # ------------------------------------------------------------------
    ref_fake = exact_attention(q_fp, k_fp, v_fp, KEY_MASKS)
    ref_gauss = exact_attention(q_gauss, k_gauss, v_gauss, KEY_MASKS)
    ref_fake_fp16 = np.array([g.float_to_fp16_hex(x) for x in ref_fake],
                             dtype=np.uint16)
    ref_gauss_fp16 = np.array([g.float_to_fp16_hex(x) for x in ref_gauss],
                              dtype=np.uint16)

    # ------------------------------------------------------------------
    # CSV: inputs, packets, scores, P blocks, output, errors.
    # ------------------------------------------------------------------
    write_csv_matrix(CSV_DIR / "q_matrix_fp.csv", "Index", "Q",
                     [f"E{i}" for i in range(16)],
                     q_gauss.reshape(1, -1))
    write_csv_matrix(CSV_DIR / "q_matrix_quantized.csv", "Index", "Q",
                     [f"E{i}" for i in range(16)],
                     np.asarray(q_fp, dtype=float).reshape(1, -1))
    write_csv_matrix(CSV_DIR / "k_matrix_fp.csv", "Key", "K",
                     [f"E{i}" for i in range(16)], k_gauss)
    write_csv_matrix(CSV_DIR / "k_matrix_quantized.csv", "Key", "K",
                     [f"E{i}" for i in range(16)], k_fp)
    write_csv_matrix(CSV_DIR / "v_matrix_fp.csv", "Out_Dim", "Dim",
                     [f"Key_{i}" for i in range(NUM_KEYS)], v_gauss)
    write_csv_matrix(CSV_DIR / "v_matrix_quantized.csv", "Out_Dim", "Dim",
                     [f"Key_{i}" for i in range(NUM_KEYS)], v_fp)

    with open(CSV_DIR / "qk_scores_q16.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block"] + [f"Key_{i}" for i in range(BLOCK_SIZE)])
        for b in range(NUM_BLOCKS):
            w.writerow([b] + [int(scores[b, i]) for i in range(BLOCK_SIZE)])

    with open(CSV_DIR / "p_blocks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "p_blk", "csum", "l_contrib", "resc_valid",
                    "resc_q", "resc_r"] + [f"K{i}_code" for i in range(16)])
        for b, blk in enumerate(p_blocks):
            w.writerow([b, blk["p_blk"], blk["csum"], blk["l_contrib"],
                        int(blk["resc_valid"]), blk["resc_q"], blk["resc_r"]] +
                       [int(x) for x in blk["codes"]])

    comparisons = [
        ("model_fp16_vs_fakequant", golden_real.reshape(-1, 1),
         np.asarray(ref_fake, dtype=float).reshape(-1, 1)),
        ("fakequant_vs_gaussian", np.asarray(ref_fake, dtype=float).reshape(-1, 1),
         np.asarray(ref_gauss, dtype=float).reshape(-1, 1)),
        ("model_fp16_vs_gaussian", golden_real.reshape(-1, 1),
         np.asarray(ref_gauss, dtype=float).reshape(-1, 1)),
    ]
    with open(CSV_DIR / "error_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "max_abs", "mean_abs", "rmse",
                    "max_rel", "mean_rel", "worst_lane", "worst_token"])
        for name, act, ref in comparisons:
            m = g.error_metrics(act, ref)
            w.writerow([name, f"{m['max_abs']:.9f}", f"{m['mean_abs']:.9f}",
                        f"{m['rmse']:.9f}", f"{m['max_rel']:.9f}",
                        f"{m['mean_rel']:.9f}", m["worst_lane"], 0])

    with open(CSV_DIR / "golden_output_combined.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lane", "o_q16", "o_hex", "l_q16", "l_hex",
                    "fp16_hex", "fp16_real", "ref_fakequant",
                    "ref_gaussian", "fp16_minus_fake", "fp16_minus_gauss"])
        for d in range(NUM_LANES):
            w.writerow([d, int(o_q16[d]), f"0x{int(o_q16[d]) & 0xFFFFFFFF:08X}",
                        int(l_q16), f"0x{int(l_q16) & 0xFFFFFFFF:08X}",
                        f"0x{int(golden_fp16[d]):04X}", f"{golden_real[d]:.9f}",
                        f"{ref_fake[d]:.9f}", f"{ref_gauss[d]:.9f}",
                        f"{golden_real[d] - ref_fake[d]:.9f}",
                        f"{golden_real[d] - ref_gauss[d]:.9f}"])

    # ------------------------------------------------------------------
    # HEX files for the TB.
    # ------------------------------------------------------------------
    def write_hex_packets(path, packets):
        with open(path, "w") as f:
            for pkt in packets:
                f.write(f"{int(pkt) & ((1 << 72) - 1):018X}\n")

    q_pkt = g.encode_72b_packet(q_codes, q_scale)
    k_pkts = []
    for key in range(NUM_KEYS):
        k_pkts.append(g.encode_72b_packet(k_codes[key], int(k_scale[key, 0])))
    v_pkts = []
    for b in range(NUM_BLOCKS):
        for d in range(NUM_LANES):
            v_pkts.append(g.encode_72b_packet(
                v_codes[d, b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE],
                int(v_scale[d, b])))
    p_pkts = []
    for blk in p_blocks:
        p_pkts.append(g.encode_72b_packet(blk["codes"], int(blk["p_blk"])))

    write_hex_packets(HEX_DIR / "q_packet.hex", [q_pkt])
    write_hex_packets(HEX_DIR / "k_packets.hex", k_pkts)
    write_hex_packets(HEX_DIR / "v_packets.hex", v_pkts)
    write_hex_packets(HEX_DIR / "golden_p_packets.hex", p_pkts)

    with open(HEX_DIR / "key_masks.hex", "w") as f:
        for m in KEY_MASKS:
            f.write(f"{m & 0xFFFF:04X}\n")

    with open(HEX_DIR / "golden_scores_q16.hex", "w") as f:
        for b in range(NUM_BLOCKS):
            for i in range(BLOCK_SIZE):
                f.write(f"{int(scores[b, i]) & 0xFFFFFFFF:08X}\n")

    with open(HEX_DIR / "golden_l_q16.hex", "w") as f:
        f.write(f"{int(l_q16) & 0xFFFFFFFF:08X}\n")

    with open(HEX_DIR / "golden_o_q16.hex", "w") as f:
        for d in range(NUM_LANES):
            f.write(f"{int(o_q16[d]) & 0xFFFFFFFF:08X}\n")

    with open(HEX_DIR / "golden_fp16.hex", "w") as f:
        for d in range(NUM_LANES):
            f.write(f"{int(golden_fp16[d]):04X}\n")

    with open(HEX_DIR / "golden_ref_fp16.hex", "w") as f:
        for d in range(NUM_LANES):
            f.write(f"{int(ref_fake_fp16[d]):04X}\n")

    with open(HEX_DIR / "golden_gauss_fp16.hex", "w") as f:
        for d in range(NUM_LANES):
            f.write(f"{int(ref_gauss_fp16[d]):04X}\n")

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print("=== Attention data (1 query, 32 keys, head dim 16) ===")
    print(f"Q scale={q_scale}, K scale range={int(k_scale.min())}..{int(k_scale.max())}, "
          f"V scale range={int(v_scale.min())}..{int(v_scale.max())}")
    print("QK scores (Q16.16):")
    for b in range(NUM_BLOCKS):
        print("  block", b, [int(x) for x in scores[b]])
    print("Softmax blocks:")
    for b, blk in enumerate(p_blocks):
        print(f"  block {b}: m={blk['m_new']}, p_blk={blk['p_blk']}, "
              f"csum={blk['csum']}, l_contrib={blk['l_contrib']}, "
              f"resc={blk['resc_valid']} (q={blk['resc_q']}, r={blk['resc_r']})")
    print(f"l_q16={l_q16} ({l_q16 / 65536.0:.9f})")
    print("O Q16.16:", [int(x) for x in o_q16])
    print("Golden FP16:", [f"0x{int(x):04X}" for x in golden_fp16])
    print("Fake-quant float:", [f"{x:.6f}" for x in ref_fake])
    print("Gaussian float  :", [f"{x:.6f}" for x in ref_gauss])
    print("Error summary:")
    for name, act, ref in comparisons:
        m = g.error_metrics(act, ref)
        print(f"  {name}: max_abs={m['max_abs']:.6g}, "
              f"mean_abs={m['mean_abs']:.6g}, rmse={m['rmse']:.6g}, "
              f"max_rel={m['max_rel']:.6g}, worst_lane={m['worst_lane']}")
    print(f"[SUCCESS] Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
