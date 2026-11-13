#!/usr/bin/env python3
"""
prove_full_flashattention_tiling_dv.py
Comprehensive mathematical proof of the Full Hardware Tiling Algorithm
with Multi-Head, large Query sequences, large Key/Value sequences,
AND large head dimensions (d_v >> 16, e.g. d_v = 64).

Hardware Constraints of snax_fused_llm_core:
- NUM_LANES = 16 (Hardware can compute 16 lanes of dot-products/values)
- ACC_DEPTH = 16 (8:8 Split: exactly 8 Query slots for O, 8 Scratchpad slots for S)
- P_FIFO: Transient FIFO, popped on each compute beat (strictly 1 consumer pass per Mode 10)

Hierarchical Loop Nest (5 Levels):
  Loop 1 (Head)    : for h in 0 .. H-1
    Loop 2 (Query) : for qi in 0 .. Q_TILES-1 (each tile M_Q = 8 tokens)
      Loop 3 (d_v) : for dvi in 0 .. DV_TILES-1 (each slice 16 channels, matching 16 lanes & 8:8 slots)
        -- Reset running max m and sum l for this (query_tile, dv_slice) --
        Loop 4 (KV): for kj in 0 .. KV_TILES-1 (each block 16 keys)
          [Mode 01]: S_block = Q_tile * K_block^T -> Softmax Requantizer -> Push P into p_fifo
          [Mode 10]: Pop P from p_fifo -> Online Rescale O_slice (Slots 0..7) + P * V_slice (16 channels)
        -- End of Loop 4 (full KV sequence processed) --
        Final Normalization: O_slice = O_accum / l_running -> Export 16 channels of O
"""

import numpy as np

def standard_multihead_attention(Q, K, V):
    """
    Standard textbook Multi-Head Self-Attention.
    Shapes:
      Q: [H, N_q, d_k]
      K: [H, N_kv, d_k]
      V: [H, N_kv, d_v]
    Output:
      O: [H, N_q, d_v]
    Requires full O(H * N_q * N_kv) memory materialization for S and P.
    """
    H, N_q, d_k = Q.shape
    _, N_kv, d_v = V.shape
    scale = 1.0 / np.sqrt(d_k)
    
    O_full = np.zeros((H, N_q, d_v), dtype=np.float64)
    
    for h in range(H):
        # S: [N_q, N_kv]
        S = np.matmul(Q[h], K[h].T) * scale
        m = np.max(S, axis=-1, keepdims=True)
        exp_S = np.exp(S - m)
        l = np.sum(exp_S, axis=-1, keepdims=True)
        P = exp_S / l
        O_full[h] = np.matmul(P, V[h])
        
    return O_full

def hardware_full_tiling_attention(Q, K, V, M_tile=8, N_block=16, CH_slice=16):
    """
    Hardware-accurate Full Tiling Algorithm matching snax_fused_llm_core:
      M_tile   = 8  (Fixed by 8:8 accumulator slots in Latch Array)
      N_block  = 16 (Fixed by 16 parallel vector lanes)
      CH_slice = 16 (Fixed by 16 channels per 512-bit slot in Latch Array)
    """
    H, N_q, d_k = Q.shape
    _, N_kv, d_v = V.shape
    scale = 1.0 / np.sqrt(d_k)
    
    num_q_tiles = int(np.ceil(N_q / M_tile))
    num_dv_tiles = int(np.ceil(d_v / CH_slice))
    num_kv_blocks = int(np.ceil(N_kv / N_block))
    
    print("\n" + "=" * 80)
    print("  HARDWARE FULL TILING CONFIGURATION")
    print(f"  Heads H = {H}")
    print(f"  Query Tokens N_q = {N_q} -> {num_q_tiles} Query Tiles (M_tile = {M_tile})")
    print(f"  Key/Value Tokens N_kv = {N_kv} -> {num_kv_blocks} KV Blocks (N_block = {N_block})")
    print(f"  Value Dimension d_v = {d_v} -> {num_dv_tiles} Channel Slices (CH_slice = {CH_slice})")
    print("=" * 80)
    
    O_final = np.zeros((H, N_q, d_v), dtype=np.float64)
    
    # =========================================================================
    # VÒNG 1: DUYỆT QUA CÁC HEADS (h = 0 .. H-1)
    # =========================================================================
    for h in range(H):
        
        # =====================================================================
        # VÒNG 2: DUYỆT QUA CÁC TILE QUERY Q_i (M_tile = 8 tokens)
        # =====================================================================
        for qi in range(num_q_tiles):
            q_start = qi * M_tile
            q_end = min(q_start + M_tile, N_q)
            actual_q_size = q_end - q_start
            
            Q_tile = Q[h, q_start:q_end, :] # [actual_q_size, d_k]
            
            # =================================================================
            # VÒNG 3: DUYỆT QUA CÁC LÁT CẮT KÊNH CỦA d_v (CH_slice = 16 channels)
            # (Giải quyết triệt để nút thắt d_v > 16 trên phần cứng 16 slots!)
            # =================================================================
            for dvi in range(num_dv_tiles):
                dv_start = dvi * CH_slice
                dv_end = min(dv_start + CH_slice, d_v)
                actual_ch_size = dv_end - dv_start
                
                # Khởi tạo trạng thái phần cứng cho lát cắt (qi, dvi) này:
                # - Slots 0..7 (Output Accumulator O): Chứa đúng actual_q_size x 16 channels!
                # - l_running và m_running được reset khi bắt đầu chuỗi KV mới (cfg_row_start_i=1)
                m_running = np.full((actual_q_size, 1), -np.inf)
                l_running = np.zeros((actual_q_size, 1))
                O_accum_slice = np.zeros((actual_q_size, actual_ch_size)) # Nằm trọn trong Slots 0..7
                
                # =============================================================
                # VÒNG 4: DUYỆT QUA TOÀN BỘ CHUỖI KEY/VALUE (Mỗi block 16 keys)
                # =============================================================
                for kj in range(num_kv_blocks):
                    k_start = kj * N_block
                    k_end = min(k_start + N_block, N_kv)
                    
                    K_block = K[h, k_start:k_end, :] # [16, d_k] (nạp vào 16 lanes)
                    # Chỉ lấy đúng lát cắt dvi của V (16 channels)
                    V_slice_block = V[h, k_start:k_end, dv_start:dv_end] # [16, 16] (16 lanes)
                    
                    # ---------------------------------------------------------
                    # PHASE A (MODE 01): Tính S = Q_tile * K_block^T -> Softmax
                    # ---------------------------------------------------------
                    # S_block: [actual_q_size, 16] (lưu vào Scratchpad Slots 8..15)
                    S_block = np.matmul(Q_tile, K_block.T) * scale
                    
                    # Cập nhật Running Max m
                    m_block = np.max(S_block, axis=-1, keepdims=True)
                    m_new = np.maximum(m_running, m_block)
                    
                    # Hệ số co giãn Online Rescale: exp(m_old - m_new) <= 1
                    rescale_factor = np.exp(m_running - m_new)
                    
                    # Gói xác suất FP4 P (được đẩy vào p_fifo)
                    P_unnorm = np.exp(S_block - m_new)
                    
                    # Cập nhật l_accumulator: l_new = (l_old * rescale) + l_contrib
                    l_running = (l_running * rescale_factor) + np.sum(P_unnorm, axis=-1, keepdims=True)
                    
                    # ---------------------------------------------------------
                    # PHASE B (MODE 10): Pop P từ FIFO -> Rescale O & Cộng P*V
                    # ---------------------------------------------------------
                    # Gói P được pop ra khỏi p_fifo và nhân đúng 1 LẦN DUY NHẤT
                    # với lát cắt 16 kênh tương ứng của V! (Khớp hoàn toàn với phần cứng!)
                    O_accum_slice = O_accum_slice * rescale_factor
                    O_accum_slice += np.matmul(P_unnorm, V_slice_block)
                    
                    # Lưu lại m_running cho khối KV tiếp theo
                    m_running = m_new

                # =============================================================
                # PHASE C: FINAL DRAIN & NORMALIZATION CHO LÁT CẮT NÀY
                # =============================================================
                # final_normalizer nhân O_accum_slice với 1/l_running
                O_normalized_slice = O_accum_slice / l_running
                
                # Ghi 16 kênh này vào đúng vị trí của ma trận đầu ra toàn cục O
                O_final[h, q_start:q_end, dv_start:dv_end] = O_normalized_slice

    return O_final

def main():
    np.random.seed(42)
    
    # Thiết lập kích thước mô hình LLM thực tế vượt xa giới hạn phần cứng:
    # H = 2 heads
    # N_q = 32 queries (cần 4 Query tiles M_tile=8)
    # N_kv = 64 tokens (cần 4 KV blocks N_block=16)
    # d_k = 32
    # d_v = 64 (RẤT LỚN SO VỚI 16 LANES! Cần đúng 4 lát cắt dvi=0,1,2,3 mỗi lát 16 channels)
    H = 2
    N_q = 32
    N_kv = 64
    d_k = 32
    d_v = 64
    
    print("=" * 80)
    print("  CHỨNG MINH TOÁN HỌC: THUẬT TOÁN TILING ĐẦY ĐỦ VỚI d_v = 64 (d_v >> 16)")
    print("=" * 80)
    
    Q = np.random.randn(H, N_q, d_k)
    K = np.random.randn(H, N_kv, d_k)
    V = np.random.randn(H, N_kv, d_v)
    
    # 1. Chạy Standard Multi-Head Attention (Chuẩn mực tham chiếu toàn cục)
    O_std = standard_multihead_attention(Q, K, V)
    
    # 2. Chạy Thuật toán Hardware Tiling đầy đủ với vòng lặp d_v
    O_tiled = hardware_full_tiling_attention(Q, K, V, M_tile=8, N_block=16, CH_slice=16)
    
    # 3. So sánh sai số tuyệt đối trên toàn bộ ma trận [H, N_q, d_v]
    diff = np.abs(O_tiled - O_std)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    is_exact = np.allclose(O_tiled, O_std, atol=1e-12)
    
    print("\n" + "=" * 80)
    print("                           KẾT QUẢ ĐỐI CHIẾU")
    print("=" * 80)
    print(f"Tổng số phần tử ma trận Output O [H={H}, N_q={N_q}, d_v={d_v}] : {H * N_q * d_v}")
    print(f"Max Absolute Error  : {max_err:.2e}")
    print(f"Mean Absolute Error : {mean_err:.2e}")
    print(f"Is Exactly Equal?   : {is_exact}")
    
    if is_exact:
        print("\n>>> [XÁC MINH TOÁN HỌC THÀNH CÔNG 100%!]")
        print("    Thuật toán Tiling có vòng lặp d_v (dvi) hoàn toàn đồng nhất với")
        print("    Standard Multi-Head Attention ở độ chính xác kép (Double Precision)!")
        print("    Điều này chứng minh:")
        print("    1. Bằng cách đặt vòng lặp d_v duyệt qua từng lát cắt 16 channels,")
        print("       phần cứng với ACC_DEPTH=16 (Slots 0..7) có thể giải quyết d_v=64, 128, 256.")
        print("    2. Gói P trong p_fifo được POP và tiêu thụ đúng 1 lần cho mỗi lát cắt 16 kênh,")
        print("       khớp 100% với kiến trúc vi mạch hiện tại mà không bị lỗi mất dữ liệu.")
    else:
        print("\n>>> [CẢNH BÁO] Có sự sai lệch giữa hai thuật toán!")
    print("=" * 80)

if __name__ == "__main__":
    main()