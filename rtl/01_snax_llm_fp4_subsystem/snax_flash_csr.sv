`timescale 1ns/1ps

module snax_flash_csr #(
    parameter int unsigned ADDR_WIDTH = 12,
    parameter int unsigned DATA_WIDTH = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    csr_we,
    input  logic [ADDR_WIDTH-1:0]   csr_addr,
    input  logic [DATA_WIDTH-1:0]   csr_wdata,
    output logic [DATA_WIDTH-1:0]   csr_rdata,

    input  logic                    seq_busy,
    input  logic                    seq_done,
    input  logic                    act_busy,
    input  logic                    wt_busy,
    input  logic                    wo_busy,
    input  logic                    v_busy,
    input  logic                    v_done,
    input  logic                    rd_busy,
    input  logic                    rd_done,
    input  logic                    p_fifo_full,
    input  logic                    p_fifo_empty,
    input  logic                    l_ready,
    input  logic                    core_fp16_valid,

    output logic                    start_pulse,
    output logic                    src_sel,
    output logic                    act_ready_force,
    output logic                    wt_ready_force,

    output logic                    act_start,
    output logic [31:0]             act_base,
    output logic [31:0]             act_k_blocks,
    output logic [31:0]             act_k_stride,
    output logic [31:0]             act_tokens,
    output logic [31:0]             act_token_stride,

    output logic                    wt_start,
    output logic [31:0]             wt_base,
    output logic [31:0]             wt_repeat,
    output logic [31:0]             wt_k_blocks,
    output logic [31:0]             wt_k_stride,
    output logic [31:0]             wt_num_tiles,
    output logic [31:0]             wt_tile_step,

    output logic                    wo_start,
    output logic                    wo_mode_fp16,
    output logic [31:0]             wo_base,
    output logic [31:0]             wo_num_vectors,
    output logic [31:0]             wo_line_stride,

    output logic [1:0]              core_mode,
    output logic                    core_row_start,
    output logic [15:0]             core_key_mask,
    output logic                    core_enable_slq,
    output logic                    core_tile_drain_req,
    output logic [3:0]              core_tile_drain_idx,
    output logic                    core_tile_clear_on_drain,
    output logic [15:0]             core_lane_valid,
    output logic [3:0]              core_lane_acc_idx,

    output logic                    v_start,
    output logic [31:0]             v_base,
    output logic [31:0]             v_num_batches,

    output logic                    rd_start,
    output logic [31:0]             rd_base,
    output logic [31:0]             rd_num_vectors,
    output logic [31:0]             rd_line_stride,

    // log-quantizer input selection
    //   2'b00 : core FP16 row   (GEMM / attention row)
    //   2'b01 : transpose column (V path)
    //   2'b10 : TCDM FP16 readback
    output logic [1:0]              quant_src_sel
);

    localparam logic [31:0] ID_VALUE = 32'h534E_4146;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse           <= 1'b0;
            src_sel               <= 1'b0;
            act_ready_force       <= 1'b1;
            wt_ready_force        <= 1'b1;
            act_start             <= 1'b0;
            act_base              <= 32'h0;
            act_k_blocks          <= 32'h0;
            act_k_stride          <= 32'h0;
            act_tokens            <= 32'h0;
            act_token_stride      <= 32'h0;
            wt_start              <= 1'b0;
            wt_base               <= 32'h0;
            wt_repeat             <= 32'h0;
            wt_k_blocks           <= 32'h0;
            wt_k_stride           <= 32'h0;
            wt_num_tiles          <= 32'h1;
            wt_tile_step          <= 32'h0;
            wo_start              <= 1'b0;
            wo_mode_fp16          <= 1'b0;
            wo_base               <= 32'h0000_0110;
            wo_num_vectors        <= 32'h0;
            wo_line_stride        <= 32'h200;
            core_mode             <= 2'b00;
            core_row_start        <= 1'b0;
            core_key_mask         <= 16'hFFFF;
            core_enable_slq       <= 1'b0;
            core_tile_drain_req   <= 1'b0;
            core_tile_drain_idx   <= 4'h0;
            core_tile_clear_on_drain <= 1'b0;
            core_lane_valid       <= 16'h0000;
            core_lane_acc_idx     <= 4'h0;
            v_start               <= 1'b0;
            v_base                <= 32'h0;
            v_num_batches         <= 32'h0;
            rd_start              <= 1'b0;
            rd_base               <= 32'h0;
            rd_num_vectors        <= 32'h0;
            rd_line_stride        <= 32'h200;
            quant_src_sel         <= 2'b00;
        end else begin
            start_pulse         <= 1'b0;
            act_start           <= 1'b0;
            wt_start            <= 1'b0;
            wo_start            <= 1'b0;
            v_start             <= 1'b0;
            rd_start            <= 1'b0;

            if (csr_we) begin
                case (csr_addr)
                    12'h000: if (csr_wdata[0]) start_pulse <= 1'b1;
                    12'h00C: begin
                        src_sel         <= csr_wdata[0];
                        act_ready_force <= csr_wdata[1];
                        wt_ready_force  <= csr_wdata[2];
                    end
                    12'h010: act_start          <= csr_wdata[0];
                    12'h014: act_base           <= csr_wdata;
                    12'h018: act_k_blocks       <= csr_wdata;
                    12'h01C: act_k_stride       <= csr_wdata;
                    12'h020: act_tokens         <= csr_wdata;
                    12'h024: act_token_stride   <= csr_wdata;
                    12'h030: wt_start           <= csr_wdata[0];
                    12'h034: wt_base            <= csr_wdata;
                    12'h038: wt_repeat          <= csr_wdata;
                    12'h03C: wt_k_blocks        <= csr_wdata;
                    12'h040: wt_k_stride        <= csr_wdata;
                    12'h044: wt_num_tiles       <= csr_wdata;
                    12'h048: wt_tile_step       <= csr_wdata;
                    12'h050: wo_start           <= csr_wdata[0];
                    12'h054: wo_base            <= csr_wdata;
                    12'h058: wo_num_vectors     <= csr_wdata;
                    12'h05C: wo_line_stride     <= csr_wdata;
                    12'h060: wo_mode_fp16       <= csr_wdata[0];
                    12'h070: core_mode          <= csr_wdata[1:0];
                    12'h074: core_row_start     <= csr_wdata[0];
                    12'h078: core_key_mask      <= csr_wdata[15:0];
                    12'h07C: core_enable_slq    <= csr_wdata[0];
                    12'h080: core_tile_drain_req <= csr_wdata[0];
                    12'h084: core_tile_drain_idx <= csr_wdata[3:0];
                    12'h088: core_tile_clear_on_drain <= csr_wdata[0];
                    12'h08C: core_lane_valid    <= csr_wdata[15:0];
                    12'h090: core_lane_acc_idx  <= csr_wdata[3:0];
                    12'h0A0: v_start            <= csr_wdata[0];
                    12'h0A4: v_base             <= csr_wdata;
                    12'h0A8: v_num_batches      <= csr_wdata;
                    12'h0B0: rd_start            <= csr_wdata[0];
                    12'h0B4: rd_base             <= csr_wdata;
                    12'h0B8: rd_num_vectors      <= csr_wdata;
                    12'h0BC: rd_line_stride      <= csr_wdata;
                    12'h0C4: quant_src_sel       <= csr_wdata[1:0];
                    default: ;
                endcase
            end
        end
    end

    always_comb begin
        case (csr_addr)
            12'h000: csr_rdata = 32'd0;
            12'h004: csr_rdata = {30'd0, (seq_done | v_done | rd_done),
                                           (seq_busy | v_busy | rd_busy)};
            12'h008: csr_rdata = ID_VALUE;
            12'h00C: csr_rdata = {29'd0, wt_ready_force, act_ready_force, src_sel};
            12'h010: csr_rdata = {31'd0, act_start};
            12'h014: csr_rdata = act_base;
            12'h018: csr_rdata = act_k_blocks;
            12'h01C: csr_rdata = act_k_stride;
            12'h020: csr_rdata = act_tokens;
            12'h024: csr_rdata = act_token_stride;
            12'h028: csr_rdata = {31'd0, act_busy};
            12'h030: csr_rdata = {31'd0, wt_start};
            12'h034: csr_rdata = wt_base;
            12'h038: csr_rdata = wt_repeat;
            12'h03C: csr_rdata = wt_k_blocks;
            12'h040: csr_rdata = wt_k_stride;
            12'h044: csr_rdata = wt_num_tiles;
            12'h048: csr_rdata = wt_tile_step;
            12'h04C: csr_rdata = {31'd0, wt_busy};
            12'h050: csr_rdata = {31'd0, wo_start};
            12'h054: csr_rdata = wo_base;
            12'h058: csr_rdata = wo_num_vectors;
            12'h05C: csr_rdata = wo_line_stride;
            12'h060: csr_rdata = {31'd0, wo_mode_fp16};
            12'h064: csr_rdata = {31'd0, wo_busy};
            12'h070: csr_rdata = {30'd0, core_mode};
            12'h074: csr_rdata = {31'd0, core_row_start};
            12'h078: csr_rdata = {16'd0, core_key_mask};
            12'h07C: csr_rdata = {31'd0, core_enable_slq};
            12'h080: csr_rdata = {31'd0, core_tile_drain_req};
            12'h084: csr_rdata = {28'd0, core_tile_drain_idx};
            12'h088: csr_rdata = {31'd0, core_tile_clear_on_drain};
            12'h08C: csr_rdata = {16'd0, core_lane_valid};
            12'h090: csr_rdata = {28'd0, core_lane_acc_idx};
            12'h094: csr_rdata = {28'd0, core_fp16_valid, l_ready,
                                  p_fifo_empty, p_fifo_full};
            12'h0A0: csr_rdata = {31'd0, v_start};
            12'h0A4: csr_rdata = v_base;
            12'h0A8: csr_rdata = v_num_batches;
            12'h0AC: csr_rdata = {30'd0, v_done, v_busy};
            12'h0B0: csr_rdata = {31'd0, rd_start};
            12'h0B4: csr_rdata = rd_base;
            12'h0B8: csr_rdata = rd_num_vectors;
            12'h0BC: csr_rdata = rd_line_stride;
            12'h0C0: csr_rdata = {30'd0, rd_done, rd_busy};
            12'h0C4: csr_rdata = {30'd0, quant_src_sel};
            default: csr_rdata = 32'd0;
        endcase
    end

endmodule
