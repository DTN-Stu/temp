`timescale 1ns/1ps

// =============================================================================
// Module: final_normalizer
// Description: Multiplies unnormalized accumulator vector O_unnorm by inv_l = 1/l
//              and converts the 16 lanes into IEEE 754 FP16 (256-bit SIMD vector).
//              Includes bypass mode for standard GeMM operations.
// =============================================================================
module final_normalizer #(
    parameter int NUM_LANES_P = 16,
    parameter int FRAC_BITS   = 16
)(
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 drain_req,
    input  logic                 bypass_inv_l,   // 1: Standard GeMM (skip 1/l); 0: Attention (multiply 1/l)
    input  logic signed [NUM_LANES_P-1:0][31:0] accum_in,
    input  logic [15:0]          inv_l_mant,     // Reciprocal mantissa (Q15)
    input  logic [4:0]           inv_l_exp,      // Exponent from LZC

    output logic                 vec_valid,
    output logic [NUM_LANES_P-1:0][15:0] vec_out,
    output logic [NUM_LANES_P*16-1:0] vec_packed
);

    int fn_l;

    logic signed [NUM_LANES_P-1:0][31:0] norm_input;
    logic [NUM_LANES_P-1:0][15:0]        fp16_lanes;

    // Stage 1 Pipeline Registers (Multiplier + Saturation Result)
    logic signed [NUM_LANES_P-1:0][31:0] s1_norm_data_q;
    logic                                s1_valid_q;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_LANE_NORM
            // Stage 1 Multiplier: 32-bit signed * 16-bit unsigned
            logic signed [47:0] raw_mult;
            logic signed [63:0] raw_mult_64;
            logic signed [63:0] scaled_wide;
            logic signed [31:0] scaled_div_l;

            always_comb begin
                raw_mult = $signed(accum_in[i]) * $signed({1'b0, inv_l_mant});
                raw_mult_64 = $signed(raw_mult);

                // inv_l_mant = 2^15/(1+f), inv_l_exp = leading-zero count (lzc).
                // 1/l = inv_l_mant * 2^(lzc-29).  For a Q16.16 accumulator
                // this is a right shift by (29-lzc), or a left shift if the
                // row denominator is very small (lzc > 29).
                if (inv_l_exp <= 5'd29)
                    scaled_wide = raw_mult_64 >>> (5'd29 - inv_l_exp);
                else
                    scaled_wide = raw_mult_64 <<< (inv_l_exp - 5'd29);

                // Saturate to the 32-bit Q16.16 accumulator range.
                if (scaled_wide > 64'sd2147483647)
                    scaled_div_l = 32'sh7FFF_FFFF;
                else if (scaled_wide < -64'sd2147483648)
                    scaled_div_l = 32'sh8000_0000;
                else
                    scaled_div_l = scaled_wide[31:0];
            end

            assign norm_input[i] = bypass_inv_l ? accum_in[i] : scaled_div_l;

            // Stage 2: IEEE 754 FP16 Conversion from Stage 1 registered data
            logscale_lazy_drain_norm #(
                .FRAC_BITS (FRAC_BITS)
            ) u_drain_norm (
                .norm_in  (s1_norm_data_q[i]),
                .fp16_out (fp16_lanes[i])
            );

            assign vec_packed[i*16 +: 16] = vec_out[i];
        end
    endgenerate

    // Stage 1 Pipeline Register Update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid_q     <= 1'b0;
            s1_norm_data_q <= '0;
        end else begin
            s1_valid_q <= drain_req;
            if (drain_req) begin
                s1_norm_data_q <= norm_input;
            end
        end
    end

    // Stage 2 Output Registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vec_valid <= 1'b0;
            for (fn_l = 0; fn_l < NUM_LANES_P; fn_l++) begin
                vec_out[fn_l] <= 16'h0000;
            end
        end else begin
            vec_valid <= s1_valid_q;
            if (s1_valid_q) begin
                for (fn_l = 0; fn_l < NUM_LANES_P; fn_l++) begin
                    vec_out[fn_l] <= fp16_lanes[fn_l];
                end
            end
        end
    end

endmodule
