#!/usr/bin/env python3
"""
prove_flashattention_tiling.py
Demonstration and mathematical equivalence proof between:
1. Standard Self-Attention (Materializes full NxN matrices S and P)
2. Hardware FlashAttention Loop Nest (Tiled on-chip with Online Rescaling)
   matching the snax_fused_llm_core hardware pipeline (NUM_LANES=16, ACC_DEPTH=16).
"""

import numpy as np

def standard_attention(Q, K, V):
    """
    Standard textbook Self-Attention:
      S = Q * K^T / sqrt(d_k)
      P = Softmax(S)
      O = P * V
    Requires O(N^2) memory for full S and P matrices.
    """
    N_q, d_k = Q.shape
    N_kv, d_v = V.shape
    
    # 1. Full S matrix: [N_q, N_kv]
    scale = 1.0 / np.sqrt(d_k)
    S = np.matmul(Q, K.T) * scale
    
    # 2. Row-wise Softmax over full sequence
    m = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - m)
    l = np.sum(exp_S, axis=-1, keepdims=True)
    P = exp_S / l
    
    # 3. Output O: [N_q, d_v]
    O = np.matmul(P, V)
    return O, P, S

def flashattention_loop_nest_hardware(Q, K, V, M_tile=8, N_block=16):
    """
    Hardware-accurate FlashAttention Loop Nest:
    - Outer Loop 1 (qi): Iterate over Query tiles (M_tile = 8 tokens, mapped to Slots 0..7)
    - Inner Loop 2 (kj): Iterate over Key/Value blocks (N_block = 16 keys, mapped to 16 lanes)
      * Mode 01: S_block = Q_tile * K_block^T, compute local max, exp, update running max
      * Mode 10: Online Rescale accumulator O, add P_block * V_block
    - Final Drain: Divide unnormalized O by running sum l (simulating final_normalizer)
    
    Memory footprint: O(M_tile * d_v) = O(1) wrt sequence length N_kv!
    """
    N_q, d_k = Q.shape
    N_kv, d_v = V.shape
    scale = 1.0 / np.sqrt(d_k)
    
    O_final = np.zeros((N_q, d_v), dtype=np.float64)
    
    num_q_tiles = int(np.ceil(N_q / M_tile))
    num_kv_blocks = int(np.ceil(N_kv / N_block))
    
    print(f"--- Running FlashAttention Loop Nest ---")
    print(f"Total Query tokens: {N_q} -> {num_q_tiles} Query tiles (M_tile={M_tile})")
    print(f"Total Key/Value tokens: {N_kv} -> {num_kv_blocks} KV blocks (N_block={N_block})")
    
    # =========================================================================
    # VÒNG LẶP NGOÀI 1: Duyệt qua các TILE QUERY Q_i
    # =========================================================================
    for qi in range(num_q_tiles):
        q_start = qi * M_tile
        q_end = min(q_start + M_tile, N_q)
        actual_q_size = q_end - q_start
        
        Q_tile = Q[q_start:q_end, :] # [M_tile, d_k]
        
        # Hardware state initialization at cfg_row_start_i = 1:
        # l_accumulator initialized to 0, running max m initialized to -inf
        # Output accumulator O in latch_array initialized to 0
        m_running = np.full((actual_q_size, 1), -np.inf) # [actual_q_size, 1]
        l_running = np.zeros((actual_q_size, 1))           # [actual_q_size, 1]
        O_accum = np.zeros((actual_q_size, d_v))           # [actual_q_size, d_v] (Slots 0..7)
        
        # =====================================================================
        # VÒNG LẶP TRONG 2: Duyệt qua TOÀN BỘ CHUỖI Key/Value K_j, V_j
        # =====================================================================
        for kj in range(num_kv_blocks):
            k_start = kj * N_block
            k_end = min(k_start + N_block, N_kv)
            
            K_block = K[k_start:k_end, :] # [16, d_k] (nạp vào 16 lanes)
            V_block = V[k_start:k_end, :] # [16, d_v] (nạp vào 16 lanes)
            
            # -----------------------------------------------------------------
            # PHASE 1 (MODE 01): Tính S_block = Q_tile * K_block^T -> Softmax
            # -----------------------------------------------------------------
            # S_block: [M_tile, 16] (lưu vào Scratchpad Slots 8..15)
            S_block = np.matmul(Q_tile, K_block.T) * scale
            
            # Tìm max cục bộ của block hiện tại
            m_block = np.max(S_block, axis=-1, keepdims=True)
            
            # Cực đại mới sau khi xem xét block này
            m_new = np.maximum(m_running, m_block)
            
            # Tính độ lệch co giãn: Delta m = m_old - m_new <= 0
            # (Phần cứng tách thành resc_q và resc_r để đưa vào vsq_rescaler_32b)
            rescale_factor = np.exp(m_running - m_new)
            
            # Số mũ không chuẩn hóa của block hiện tại: exp(S_block - m_new)
            P_unnorm = np.exp(S_block - m_new) # [M_tile, 16] (đẩy vào p_fifo)
            
            # Đóng góp mẫu số l của block này
            l_block_contrib = np.sum(P_unnorm, axis=-1, keepdims=True)
            
            # Cập nhật l_accumulator: l_new = (l_old * rescale) + l_contrib
            l_running = (l_running * rescale_factor) + l_block_contrib
            
            # -----------------------------------------------------------------
            # PHASE 2 (MODE 10): Tích lũy P * V và Online Rescaling O
            # -----------------------------------------------------------------
            # Bước A: Co giãn giá trị cũ của O đang lưu trong Slots 0..7:
            #   O_rescaled = O_old * 2^(m_old - m_new)
            O_accum = O_accum * rescale_factor
            
            # Bước B: Cộng dồn tích mới P_unnorm * V_block vào Slots 0..7
            O_accum += np.matmul(P_unnorm, V_block)
            
            # Cập nhật running max m cho chu kỳ tiếp theo
            m_running = m_new

        # =====================================================================
        # PHASE 3: FINAL NORMALIZATION (Tương ứng khối final_normalizer)
        # =====================================================================
        # Sau khi duyệt hết toàn bộ chuỗi K, V: O_tile = O_accum / l_running
        O_final[q_start:q_end, :] = O_accum / l_running

    return O_final

def main():
    np.random.seed(2024)
    
    # Kích thước bài toán thực tế
    # N_seq = 64 tokens (duyệt qua 4 blocks 16 keys), d_k = d_v = 32, M_tile = 8
    N_seq = 64
    d_k = 32
    d_v = 32
    M_tile = 8
    N_block = 16
    
    print("=" * 75)
    print("  CHỨNG MINH TOÁN HỌC: FLASHATTENTION LOOP NEST vs STANDARD ATTENTION")
    print(f"  Sequence Length N = {N_seq}, Head Dim d_k = {d_k}, d_v = {d_v}")
    print(f"  Hardware Tile: M_tile = {M_tile} (Query Slots), N_block = {N_block} (Hardware Lanes)")
    print("=" * 75)
    
    # Khởi tạo ngẫu nhiên Q, K, V
    Q = np.random.randn(N_seq, d_k)
    K = np.random.randn(N_seq, d_k)
    V = np.random.randn(N_seq, d_v)
    
    # 1. Chạy Standard Attention (Chuẩn mực toán học tham chiếu)
    O_std, P_std, S_std = standard_attention(Q, K, V)
    
    # 2. Chạy FlashAttention Loop Nest (Mô hình hóa phần cứng RTL)
    O_flash = flashattention_loop_nest_hardware(Q, K, V, M_tile=M_tile, N_block=N_block)
    
    # 3. So sánh sai số tuyệt đối và tương đối
    abs_diff = np.abs(O_flash - O_std)
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)
    
    print("\n" + "=" * 75)
    print("                         KẾT QUẢ ĐỐI CHIẾU")
    print("=" * 75)
    print(f"Max Absolute Error  : {max_abs_err:.2e}")
    print(f"Mean Absolute Error : {mean_abs_err:.2e}")
    print(f"Is Exactly Equal?   : {np.allclose(O_flash, O_std, atol=1e-12)}")
    
    if np.allclose(O_flash, O_std, atol=1e-12):
        print("\n>>> [CHỨNG MINH THÀNH CÔNG] Thuật toán FlashAttention Loop Nest cho ra")
        print("    kết quả chính xác 100% so với Standard Attention ở độ chính xác dấu phẩy động (Double Precision)!")
        print("    Điều này chứng minh việc giữ O trong Latch Array và co giãn Online Rescaling")
        print("    là đồng nhất hoàn toàn về mặt toán học với việc tính Softmax toàn cục!")
    else:
        print("\n>>> [CẢNH BÁO] Có sự sai lệch giữa hai thuật toán!")
    print("=" * 75)

if __name__ == "__main__":
    main()
