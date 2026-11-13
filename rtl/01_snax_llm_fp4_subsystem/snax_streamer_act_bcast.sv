// =============================================================================
// Module: snax_streamer_act_bcast.sv
// Description: Clean, High-PPA Activation / Query Broadcast Read Streamer
// - Connects to TCDM Master Port 1 (72-bit Data Width)
// - Unified 2D Nested Loop AGU:
//     * Inner Loop (Loop 0): Tokens (0 .. cfg_tokens_i - 1)
//     * Outer Loop (Loop 1): K-Blocks / d_k (0 .. cfg_k_blocks_i - 1)
// - Matches Weight Streamer & Accumulator Latch Slots (0..15)
// - Canonical Elastic Skid Buffer (depth parameterised) for zero-bubble
//   handshaking
// =============================================================================
`timescale 1ns/1ps

module snax_streamer_act_bcast #(
  parameter int unsigned DATA_WIDTH = 72, // 72-bit Packet (64b FP4 + 8b Exp)
  parameter int unsigned ADDR_WIDTH = 32, // 32-bit Address
  parameter int unsigned STRB_WIDTH = 9,  // 9-bit Byte Strobe
  // Pipeline depth of the read path.  Must be >= the TCDM read round trip
  // (interconnect request register + bank register + response window), or the
  // core stalls every beat waiting for the next token.
  // 4 covers the ~3-cycle read round trip; going deeper does not help, the
  // residual stalls are the pipeline fill at each launch, not steady state.
  parameter int unsigned SKID_DEPTH = 4
)(
  input  logic                  clk,
  input  logic                  rst_n,

  // -------------------------------------------------------------------------
  // 1. CSR Control & 2D AGU Configuration
  // -------------------------------------------------------------------------
  input  logic                  csr_start_i,
  output logic                  busy_o,
  output logic                  done_irq_o,

  input  logic [ADDR_WIDTH-1:0] cfg_base_addr_i,    // Địa chỉ gốc trong TCDM (ví dụ Bank 16)
  input  logic [31:0]           cfg_tokens_i,       // Số token trong mẻ (Loop 0: Vòng lặp trong)
  input  logic [31:0]           cfg_token_stride_i, // Bước nhảy địa chỉ sang token kế (giữa các bank)
  input  logic [31:0]           cfg_k_blocks_i,     // Số block K / d_k (Loop 1: Vòng lặp ngoài)
  input  logic [31:0]           cfg_k_stride_i,     // Bước nhảy địa chỉ sang block K kế (chiều dọc)

  // -------------------------------------------------------------------------
  // 2. TCDM Interconnect Master Port 1
  // -------------------------------------------------------------------------
  output logic                  m_tcdm_req_valid_o,
  output logic                  m_tcdm_req_write_o,
  output logic [ADDR_WIDTH-1:0] m_tcdm_req_addr_o,
  output logic [DATA_WIDTH-1:0] m_tcdm_req_wdata_o,
  output logic [STRB_WIDTH-1:0] m_tcdm_req_strb_o,
  input  logic                  m_tcdm_req_ready_i,

  input  logic [DATA_WIDTH-1:0] m_tcdm_rsp_rdata_i,
  input  logic                  m_tcdm_rsp_valid_i,

  // -------------------------------------------------------------------------
  // 3. Accelerator Shell Interface (ext_act_broadcast_i/valid/ready)
  // -------------------------------------------------------------------------
  output logic [DATA_WIDTH-1:0] act_stream_data_o,
  output logic                  act_stream_valid_o,
  input  logic                  act_stream_ready_i
);

  // =========================================================================
  // 1. TỐI GIẢN FSM & 2D AGU CONTROLLER
  // =========================================================================
  typedef enum logic [1:0] { IDLE, STREAM, DRAIN_WAIT } state_t;
  state_t state_q, state_d;

  logic [31:0] tok_cnt_q, tok_cnt_d;
  logic [31:0] k_cnt_q,   k_cnt_d;
  logic [ADDR_WIDTH-1:0] curr_addr_q, curr_addr_d;
  logic [ADDR_WIDTH-1:0] k_base_addr_q, k_base_addr_d;

  // Job descriptor, sampled at start.  A producer is free to change cfg_*
  // after the launch pulse, so the AGU must not read them live.
  logic [31:0]           tokens_q,      token_stride_q;
  logic [31:0]           k_blocks_q,    k_stride_q;

  // Điều kiện biên vòng lặp
  wire tok_last = (tok_cnt_q == tokens_q - 1'b1);
  wire k_last   = (k_cnt_q   == k_blocks_q - 1'b1);

  // Bắt tay với Elastic Skid Buffer
  localparam int unsigned PTR_W   = (SKID_DEPTH > 1) ? $clog2(SKID_DEPTH) : 1;
  localparam int unsigned LEVEL_W = $clog2(SKID_DEPTH + 1);

  logic [DATA_WIDTH-1:0] skid_mem [0:SKID_DEPTH-1];
  logic [PTR_W-1:0]   wr_p, rd_p;
  logic [LEVEL_W-1:0] count;
  wire skid_ready = (count < LEVEL_W'(SKID_DEPTH));
  wire skid_valid = (count != '0);

  wire tcdm_fire = m_tcdm_req_valid_o && m_tcdm_req_ready_i;
  wire core_fire = act_stream_valid_o && act_stream_ready_i;

  // Outstanding-read credit: the AGU advances on request acceptance, so the
  // request stream can outrun TCDM responses.  Buffered + outstanding entries
  // must stay below SKID_DEPTH so a returning response can never overflow the
  // buffer.  The window is the same SKID_DEPTH, which lets the streamer keep
  // the whole read latency in flight instead of one token per beat.
  logic [LEVEL_W-1:0] outstanding_q, outstanding_d;
  wire req_counted = tcdm_fire && (state_q == STREAM);
  wire rsp_counted = m_tcdm_rsp_valid_i && (state_q == STREAM || state_q == DRAIN_WAIT);
  wire issue_ok    = skid_ready & ((outstanding_q + count) < LEVEL_W'(SKID_DEPTH));

  always_comb begin
    state_d       = state_q;
    tok_cnt_d     = tok_cnt_q;
    k_cnt_d       = k_cnt_q;
    curr_addr_d   = curr_addr_q;
    k_base_addr_d = k_base_addr_q;
    outstanding_d = outstanding_q;
    done_irq_o    = 1'b0;

    m_tcdm_req_valid_o = 1'b0;
    m_tcdm_req_write_o = 1'b0; // Đọc
    m_tcdm_req_addr_o  = curr_addr_q;
    m_tcdm_req_wdata_o = '0;
    m_tcdm_req_strb_o  = '0;

    case (state_q)
      IDLE: begin
        if (csr_start_i) begin
          state_d       = STREAM;
          tok_cnt_d     = '0;
          k_cnt_d       = '0;
          curr_addr_d   = cfg_base_addr_i;
          k_base_addr_d = cfg_base_addr_i;
        end
      end

      STREAM: begin
        m_tcdm_req_valid_o = issue_ok;

        if (tcdm_fire) begin
          // VÒNG LẶP TRONG: Quét lần lượt từng Token trong mẻ (ví dụ 16 tokens)
          if (!tok_last) begin
            tok_cnt_d   = tok_cnt_q + 1'b1;
            curr_addr_d = curr_addr_q + token_stride_q; // Nhảy sang Token kế tiếp
          end else begin
            tok_cnt_d = '0;
            // VÒNG LẶP NGOÀI: Khi duyệt xong mẻ token, tăng 1 block K_dim
            if (!k_last) begin
              k_cnt_d       = k_cnt_q + 1'b1;
              k_base_addr_d = k_base_addr_q + k_stride_q;
              curr_addr_d   = k_base_addr_q + k_stride_q; // Reset về token 0 của block K mới
            end else begin
              state_d = DRAIN_WAIT;
            end
          end
        end
      end

      DRAIN_WAIT: begin
        // Wait for both the skid buffer and all outstanding reads.
        if (count == '0 && outstanding_q == '0) begin
          done_irq_o = 1'b1;
          state_d    = IDLE;
        end
      end

      default: state_d = IDLE;
    endcase

    if (req_counted)
      outstanding_d = outstanding_d + 1'b1;
    if (rsp_counted && (outstanding_q != 3'd0))
      outstanding_d = outstanding_d - 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q       <= IDLE;
      tok_cnt_q     <= '0;
      k_cnt_q       <= '0;
      curr_addr_q   <= '0;
      k_base_addr_q <= '0;
      outstanding_q <= '0;
    end else begin
      state_q       <= state_d;
      tok_cnt_q     <= tok_cnt_d;
      k_cnt_q       <= k_cnt_d;
      curr_addr_q   <= curr_addr_d;
      k_base_addr_q <= k_base_addr_d;
      outstanding_q <= outstanding_d;

      if (csr_start_i) begin
        tokens_q       <= cfg_tokens_i;
        token_stride_q <= cfg_token_stride_i;
        k_blocks_q     <= cfg_k_blocks_i;
        k_stride_q     <= cfg_k_stride_i;
      end
    end
  end

  // =========================================================================
  // 2. ELASTIC SKID BUFFER (SKID_DEPTH entries)
  // =========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_p  <= '0;
      rd_p  <= '0;
      count <= '0;
    end else if (state_q == STREAM || state_q == DRAIN_WAIT) begin
      case ({m_tcdm_rsp_valid_i, core_fire})
        2'b10: begin // Push from TCDM
          skid_mem[wr_p] <= m_tcdm_rsp_rdata_i;
          wr_p           <= wr_p + 1'b1;
          count          <= count + 1'b1;
        end
        2'b01: begin // Pop to Core
          rd_p  <= rd_p + 1'b1;
          count <= count - 1'b1;
        end
        2'b11: begin // Back-to-back simultaneous Push & Pop
          skid_mem[wr_p] <= m_tcdm_rsp_rdata_i;
          wr_p           <= wr_p + 1'b1;
          rd_p           <= rd_p + 1'b1;
        end
        default: ;
      endcase
    end
  end

  // Đầu ra đi trực tiếp từ Skid Buffer, không qua Mux phụ
  assign act_stream_data_o  = skid_mem[rd_p];
  assign act_stream_valid_o = skid_valid;

  assign busy_o = (state_q != IDLE) || (count != '0);

endmodule
