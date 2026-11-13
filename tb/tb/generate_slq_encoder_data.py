#!/usr/bin/env python3
"""
generate_slq_encoder_data.py

Bit-exact golden generator for slq_encoder.sv (module log_quantizer).

RTL pipeline (2 cycles):
  in_valid at edge T
    -> stage 1 at T+1: mu_max_q1, mu_d1, sign_fp16_q1, in_valid_q1
    -> stage 2 at T+2: packed_vec, out_valid

Packet layout:
  packed_vec[4*i +: 4] = fp4 code of lane i (sign in bit 3, magnitude bits 2:0)
  packed_vec[71:64]    = scale_sat (two's complement, 8-bit)

FP16 -> mu:
  E == 0      -> mu = 0
  E == 31     -> mu = 247
  otherwise   -> mu = {E[4:0], M[9:7]}

scale:
  mu_max == 0 -> 0
  else        -> clamp(mu_max - 140, -128, 127)

code:
  diff = (mu_max - mu) & 0xff
  diff <=  2 -> mag 7 (6.0)
  diff <=  6 -> mag 6 (4.0)
  diff <= 10 -> mag 5 (3.0)
  diff <= 14 -> mag 4 (2.0)
  diff <= 18 -> mag 3 (1.5)
  diff <= 24 -> mag 2 (1.0)
  diff <= 36 -> mag 1 (0.5)
  else       -> mag 0 (0.0)
  code = 0 if mag == 0 else {sign, mag}
"""
import os
import csv
import random

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "sim", "sim_slq_encoder"))
HEX_DIR = os.path.join(OUT_DIR, "hex")
CSV_DIR = os.path.join(OUT_DIR, "csv")

SCALE_OFFSET = 140
THRESHOLDS = (2, 6, 10, 14, 18, 24, 36)
MAGS = (7, 6, 5, 4, 3, 2, 1, 0)


def fp16_bits_from_mu(mu, sign=0):
    """Build an FP16 bit pattern whose RTL mu equals `mu` when possible."""
    sign_bit = (sign & 1) << 15
    if mu == 0:
        return sign_bit
    if mu == 247:
        # E=31, M=0 -> Inf / NaN class, mapped to 247
        return sign_bit | (0x1F << 10)
    e = (mu >> 3) & 0x1F
    mtop = mu & 0x7
    if e == 0:
        # mu 1..7 is not representable; E=0 is the zero class in the RTL.
        return sign_bit
    if e == 0x1F:
        return sign_bit | (0x1F << 10)
    return sign_bit | (e << 10) | (mtop << 7)


def make_block(mus, signs=None):
    if signs is None:
        signs = [0] * 16
    assert len(mus) == 16 and len(signs) == 16
    return [fp16_bits_from_mu(mu, s) for mu, s in zip(mus, signs)]


def model_block(fp16):
    mus = []
    signs = []
    for b in fp16:
        e = (b >> 10) & 0x1F
        m = b & 0x3FF
        s = (b >> 15) & 1
        if e == 0:
            mu = 0
        elif e == 0x1F:
            mu = 247
        else:
            mu = ((e << 3) | ((m >> 7) & 0x7)) & 0xFF
        mus.append(mu)
        signs.append(s)

    mu_max = max(mus)
    if mu_max == 0:
        scale = 0
    else:
        raw = mu_max - SCALE_OFFSET
        if raw > 127:
            scale = 127
        elif raw < -128:
            scale = -128
        else:
            scale = raw

    codes = []
    mags = []
    for i, mu in enumerate(mus):
        if mu == 0 or mu_max == 0:
            mag = 0
        else:
            diff = (mu_max - mu) & 0xFF
            mag = 0
            for th, mg in zip(THRESHOLDS, MAGS[:-1]):
                if diff <= th:
                    mag = mg
                    break
        mags.append(mag)
        code = 0 if mag == 0 else ((signs[i] << 3) | mag)
        codes.append(code)

    packet = (scale & 0xFF) << 64
    for i, code in enumerate(codes):
        packet |= (code & 0xF) << (4 * i)

    return {
        "mus": mus,
        "signs": signs,
        "mu_max": mu_max,
        "scale": scale,
        "mags": mags,
        "codes": codes,
        "packet": packet,
    }


def add_case(cases, name, fp16):
    assert len(fp16) == 16
    cases.append((name, fp16))


def main():
    os.makedirs(HEX_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    rng = random.Random(0x51A9)

    cases = []

    # ---- zero / subnormal / non-finite / max normal ----
    add_case(cases, "all_zero_pos", [0x0000] * 16)
    add_case(cases, "all_zero_neg", [0x8000] * 16)
    add_case(cases, "subnormal_mix",
             [0x0001, 0x8001, 0x03FF, 0x83FF, 0x0000, 0x8000,
              0x0010, 0x8010, 0x0200, 0x8200, 0x0002, 0x8002,
              0x0100, 0x8100, 0x0003, 0x8003])
    add_case(cases, "all_inf_pos", [0x7C00] * 16)
    add_case(cases, "all_inf_neg", [0xFC00] * 16)
    add_case(cases, "all_nan", [0x7E00] * 16)
    add_case(cases, "all_max_normal", [0x7BFF] * 16)
    add_case(cases, "all_neg_max_normal", [0xFBFF] * 16)

    # ---- scale saturation: mu_max < 12 -> raw < -128 -> clamp -128 ----
    add_case(cases, "scale_clamp_m8",
             make_block([8, 8, 8, 8, 8, 8, 8, 8] + [0] * 8))
    add_case(cases, "scale_clamp_m11",
             make_block([11, 11, 11, 11, 10, 9, 8, 11] + [0] * 8))
    add_case(cases, "scale_exact_m12",
             make_block([12, 12, 12, 12, 11, 10, 9, 8] + [0] * 8))
    add_case(cases, "scale_m13",
             make_block([13, 13, 13, 12, 11, 10, 9, 8] + [0] * 8))

    # ---- threshold boundaries around mu_max=140 (scale 0) ----
    diffs_a = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    add_case(cases, "thresholds_140_a",
             make_block([140 - d for d in diffs_a]))
    diffs_b = [23, 24, 25, 35, 36, 37, 50, 100, 132,
               0, 2, 6, 10, 14, 18, 24]
    add_case(cases, "thresholds_140_b",
             make_block([140 - d for d in diffs_b]))

    # ---- threshold boundaries around mu_max=247 (scale 107) ----
    add_case(cases, "thresholds_247",
             make_block([247 - d for d in diffs_a]))
    add_case(cases, "thresholds_247_zero",
             make_block([247 - d for d in diffs_b]))

    # ---- signs: magnitude code must keep sign, zero must stay +0 ----
    signs = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    add_case(cases, "signs_mixed",
             make_block([140, 139, 138, 137, 135, 134, 133, 132,
                         131, 130, 129, 128, 127, 126, 125, 124], signs))
    add_case(cases, "signs_zero",
             make_block([0] * 8 + [140] * 8, signs))

    # ---- one-hot max positions: exercises the max tree and zero lanes ----
    for j in range(16):
        mus = [0] * 16
        sgn = [0] * 16
        mus[j] = 200
        sgn[j] = j & 1
        add_case(cases, f"one_hot_{j}", make_block(mus, sgn))

    # ---- random FP16 blocks, including E=0/31 ----
    for n in range(256):
        bits = []
        for _ in range(16):
            e = rng.randrange(0, 32)
            m = rng.randrange(0, 1024)
            s = rng.randrange(0, 2)
            bits.append((s << 15) | (e << 10) | m)
        add_case(cases, f"random_{n:03d}", bits)

    # ---- write hex + CSV ----
    in_path = os.path.join(HEX_DIR, "slq_input.hex")
    exp_path = os.path.join(HEX_DIR, "slq_expected.hex")
    cnt_path = os.path.join(HEX_DIR, "slq_count.txt")
    vec_csv = os.path.join(CSV_DIR, "slq_vectors.csv")
    case_csv = os.path.join(CSV_DIR, "slq_case_summary.csv")

    n_cases = len(cases)
    with open(in_path, "w") as fi, open(exp_path, "w") as fe:
        for name, fp16 in cases:
            vec = 0
            for i, b in enumerate(fp16):
                vec |= (b & 0xFFFF) << (16 * i)
            m = model_block(fp16)
            fi.write(f"{vec:064x}\n")
            fe.write(f"{m['packet']:018x}\n")
    with open(cnt_path, "w") as fc:
        fc.write(f"{n_cases}\n")

    with open(vec_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vector", "case", "lane", "fp16_hex", "mu", "sign",
                    "mag", "code", "scale", "packet"])
        for v, (name, fp16) in enumerate(cases):
            m = model_block(fp16)
            for i in range(16):
                w.writerow([
                    v, name, i, f"{fp16[i]:04x}", m["mus"][i], m["signs"][i],
                    m["mags"][i], m["codes"][i], m["scale"],
                    f"{m['packet']:018x}",
                ])

    with open(case_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vector", "case", "mu_max", "scale", "packet"])
        for v, (name, fp16) in enumerate(cases):
            m = model_block(fp16)
            w.writerow([v, name, m["mu_max"], m["scale"],
                        f"{m['packet']:018x}"])

    print(f"[SUCCESS] {n_cases} vectors written to {OUT_DIR}")
    print(f"  {in_path}")
    print(f"  {exp_path}")
    print(f"  {vec_csv}")


if __name__ == "__main__":
    main()
