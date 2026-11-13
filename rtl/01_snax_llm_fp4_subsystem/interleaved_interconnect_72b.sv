// =============================================================================
// Module: interleaved_interconnect_72b.sv
// Description: Multi-Master to 32-Bank Interleaved Crossbar Interconnect (DW=72)
//              - Kết nối N Masters (Host, Act Streamer, 16x Weight Channels) với 32 Banks
//              - Phân xử tranh chấp Round-Robin độc lập cho từng Bank
//              - Theo dõi phản hồi đọc theo đường ống 1 chu kỳ (Zero-Deadlock)
// =============================================================================
`timescale 1ns/1ps

module round_robin_arb_n #(
    parameter int unsigned NUM_REQS = 18
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [NUM_REQS-1:0]   req_i,
    output logic [NUM_REQS-1:0]   grant_o
);

    logic [NUM_REQS-1:0] mask_q, mask_d;
    logic [NUM_REQS-1:0] masked_req;
    logic [NUM_REQS-1:0] unmasked_grant, masked_grant;

    assign masked_req     = req_i & mask_q;
    assign masked_grant   = masked_req & ~(masked_req - 1'b1);
    assign unmasked_grant = req_i & ~(req_i - 1'b1);

    assign grant_o = (|masked_req) ? masked_grant : unmasked_grant;

    always_comb begin
        mask_d = mask_q;
        if (|grant_o) begin
            mask_d = ~(grant_o | (grant_o - 1'b1));
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mask_q <= '1;
        end else begin
            mask_q <= mask_d;
        end
    end

endmodule


module interleaved_interconnect_72b #(
    parameter int unsigned NUM_MASTERS   = 18,   // Master 0: Host/DMA, 1: Act, 2..17: 16x Weights
    parameter int unsigned HOST_MASTER   = 0,    // lowest-priority (DMA) master index
    parameter logic [NUM_MASTERS-1:0] HOST_MASK = '0, // extra low-priority masters
    parameter int unsigned NUM_BANKS     = 32,   // 32 SRAM Banks
    parameter int unsigned DATA_WIDTH    = 72,   // 72 bits
    parameter int unsigned ADDR_WIDTH    = 32,   // 32-bit Address
    parameter int unsigned BANK_DEPTH    = 1024,
    parameter int unsigned BYTE_OFFSET_W = 4,    // 16-byte alignment per 72b packet slot
    localparam int unsigned BANK_SEL_W   = $clog2(NUM_BANKS), // 5 bits
    localparam int unsigned BANK_ADDR_W  = $clog2(BANK_DEPTH),
    localparam int unsigned STRB_WIDTH   = 9
)(
    input  logic                                    clk,
    input  logic                                    rst_n,

    // -------------------------------------------------------------------------
    // Masters Interface (Host, Streamers)
    // -------------------------------------------------------------------------
    input  logic [NUM_MASTERS-1:0]                  m_req_valid_i,
    output logic [NUM_MASTERS-1:0]                  m_req_ready_o,
    input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]  m_req_addr_i,
    input  logic [NUM_MASTERS-1:0]                  m_req_write_i,
    input  logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]  m_req_wdata_i,
    input  logic [NUM_MASTERS-1:0][STRB_WIDTH-1:0]  m_req_strb_i,

    output logic [NUM_MASTERS-1:0]                  m_rsp_valid_o,
    output logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]  m_rsp_rdata_o,

    // -------------------------------------------------------------------------
    // SRAM Banks Interface (tới 32 Banks)
    // -------------------------------------------------------------------------
    output logic [NUM_BANKS-1:0]                    b_req_o,
    output logic [NUM_BANKS-1:0]                    b_write_o,
    output logic [NUM_BANKS-1:0][BANK_ADDR_W-1:0]   b_addr_o,
    output logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]    b_wdata_o,
    output logic [NUM_BANKS-1:0][STRB_WIDTH-1:0]    b_strb_o,

    input  logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]    b_rdata_i,
    input  logic [NUM_BANKS-1:0]                    b_rvalid_i
);

    // =========================================================================
    // 1. GIẢI MÃ ĐỊA CHỈ XEN KẼ (INTERLEAVED BANK DECODING)
    // =========================================================================
    logic [NUM_MASTERS-1:0][BANK_SEL_W-1:0]  m_target_bank;
    logic [NUM_MASTERS-1:0][BANK_ADDR_W-1:0] m_bank_addr;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0]   bank_req_matrix;

    genvar m, b;
    generate
        for (m = 0; m < NUM_MASTERS; m++) begin : GEN_ADDR_DECODE
            // Bóc tách 5-bit Bank Index [8:4]
            assign m_target_bank[m] = m_req_addr_i[m][BYTE_OFFSET_W +: BANK_SEL_W];
            // Địa chỉ dòng bên trong từng Bank [18:9]
            assign m_bank_addr[m]   = m_req_addr_i[m][(BYTE_OFFSET_W + BANK_SEL_W) +: BANK_ADDR_W];

            for (b = 0; b < NUM_BANKS; b++) begin : GEN_REQ_MAP
                assign bank_req_matrix[b][m] = m_req_valid_i[m] && (m_target_bank[m] == b);
            end
        end
    endgenerate

    // =========================================================================
    // 2. FIXED-PRIORITY PER-BANK ARBITRATION
    //    Stream engines (act/wt/out) always win over host/DMA on the same bank.
    // =========================================================================
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] bank_grant_matrix;
    logic [NUM_MASTERS-1:0]                master_granted;

    generate
        for (b = 0; b < NUM_BANKS; b++) begin : GEN_ARBITERS
            fixed_priority_arb_n #(
                .NUM_REQS    (NUM_MASTERS),
                .HOST_MASTER (HOST_MASTER),
                .HOST_MASK   (HOST_MASK)
            ) u_bank_arb (
                .req_i   (bank_req_matrix[b]),
                .grant_o (bank_grant_matrix[b])
            );
        end
    endgenerate

    // Tính m_req_ready_o cho từng Master
    always_comb begin
        for (int i = 0; i < NUM_MASTERS; i++) begin
            master_granted[i] = 1'b0;
            for (int j = 0; j < NUM_BANKS; j++) begin
                if (bank_grant_matrix[j][i]) begin
                    master_granted[i] = 1'b1;
                end
            end
            m_req_ready_o[i] = master_granted[i];
        end
    end

    // =========================================================================
    // 3. MUX TÍN HIỆU GHI VÀO TỪNG BANK
    // =========================================================================
    always_comb begin
        for (int j = 0; j < NUM_BANKS; j++) begin
            b_req_o[j]   = |bank_grant_matrix[j];
            b_write_o[j] = 1'b0;
            b_addr_o[j]  = '0;
            b_wdata_o[j] = '0;
            b_strb_o[j]  = '0;

            for (int i = 0; i < NUM_MASTERS; i++) begin
                if (bank_grant_matrix[j][i]) begin
                    b_write_o[j] = m_req_write_i[i];
                    b_addr_o[j]  = m_bank_addr[i];
                    b_wdata_o[j] = m_req_wdata_i[i];
                    b_strb_o[j]  = m_req_strb_i[i];
                end
            end
        end
    end

    // =========================================================================
    // 4. THEO VẾT ĐƯỜNG ỐNG PHẢN HỒI ĐỌC (PIPELINED RESPONSE TRACKER)
    // =========================================================================
    logic [NUM_BANKS-1:0][$clog2(NUM_MASTERS)-1:0] grant_master_id_q;
    logic [NUM_BANKS-1:0]                          read_in_flight_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_in_flight_q  <= '0;
            grant_master_id_q <= '0;
        end else begin
            for (int j = 0; j < NUM_BANKS; j++) begin
                read_in_flight_q[j] <= 1'b0;
                for (int i = 0; i < NUM_MASTERS; i++) begin
                    if (bank_grant_matrix[j][i] && !m_req_write_i[i]) begin
                        read_in_flight_q[j]  <= 1'b1;
                        grant_master_id_q[j] <= i;
                    end
                end
            end
        end
    end

    // Định tuyến dữ liệu đọc trả về đúng Master
    always_comb begin
        m_rsp_valid_o = '0;
        m_rsp_rdata_o = '0;

        for (int i = 0; i < NUM_MASTERS; i++) begin
            for (int j = 0; j < NUM_BANKS; j++) begin
                if (b_rvalid_i[j] && read_in_flight_q[j] && (grant_master_id_q[j] == i)) begin
                    m_rsp_valid_o[i] = 1'b1;
                    m_rsp_rdata_o[i] = b_rdata_i[j];
                end
            end
        end
    end

endmodule
