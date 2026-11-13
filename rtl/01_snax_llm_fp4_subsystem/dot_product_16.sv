`timescale 1ns/1ps

module dot_product_16 (
    input  logic [63:0]        act_vec, // 16 x 4-bit FP4 activation
    input  logic [63:0]        wt_vec,  // 16 x 4-bit FP4 weight
    output logic signed [15:0] dot_out  // 16-bit signed intra-block sum (active 13b Q2: [-2304, +2304])
);

    logic signed [15:0][8:0] mult_out;

    genvar i, b;
    generate
        for (i = 0; i < 16; i++) begin : gen_mult
            fp4_mult_lut u_fp4_mult (
                .a       (act_vec[i*4 +: 4]),
                .b       (wt_vec[i*4 +: 4]),
                .prod_out(mult_out[i])
            );
        end
    endgenerate

    // Sign-extend all 16 inputs to 13 bits (max sum = 16 * 144 = 2304 < 4096)
    logic [15:0][12:0] p_ext;
    generate
        for (i = 0; i < 16; i++) begin : gen_ext
            assign p_ext[i] = {{4{mult_out[i][8]}}, mult_out[i]};
        end
    endgenerate

    // LEVEL 1: 4 parallel 13-bit 4:2 Compressors (16 inputs -> 8 outputs)
    logic [3:0][12:0] s_l1;
    logic [3:0][12:0] c_l1;
    logic [3:0][13:0] cout_l1;

    generate
        for (i = 0; i < 4; i++) begin : gen_csa_l1
            assign cout_l1[i][0] = 1'b0;
            for (b = 0; b < 13; b++) begin : gen_bits
                compressor_4to2 u_c42_l1 (
                    .in0  (p_ext[4*i + 0][b]),
                    .in1  (p_ext[4*i + 1][b]),
                    .in2  (p_ext[4*i + 2][b]),
                    .in3  (p_ext[4*i + 3][b]),
                    .cin  (cout_l1[i][b]),
                    .sum  (s_l1[i][b]),
                    .carry(c_l1[i][b]),
                    .cout (cout_l1[i][b+1])
                );
            end
        end
    endgenerate

    logic [3:0][12:0] c_l1_shifted;
    generate
        for (i = 0; i < 4; i++) begin : gen_c_shift_l1
            assign c_l1_shifted[i] = {c_l1[i][11:0], 1'b0};
        end
    endgenerate

    // LEVEL 2: 2 parallel 13-bit 4:2 Compressors (8 inputs -> 4 outputs)
    logic [1:0][12:0] s_l2;
    logic [1:0][12:0] c_l2;
    logic [1:0][13:0] cout_l2;

    generate
        for (i = 0; i < 2; i++) begin : gen_csa_l2
            assign cout_l2[i][0] = 1'b0;
            for (b = 0; b < 13; b++) begin : gen_bits
                compressor_4to2 u_c42_l2 (
                    .in0  (s_l1[2*i + 0][b]),
                    .in1  (c_l1_shifted[2*i + 0][b]),
                    .in2  (s_l1[2*i + 1][b]),
                    .in3  (c_l1_shifted[2*i + 1][b]),
                    .cin  (cout_l2[i][b]),
                    .sum  (s_l2[i][b]),
                    .carry(c_l2[i][b]),
                    .cout (cout_l2[i][b+1])
                );
            end
        end
    endgenerate

    logic [1:0][12:0] c_l2_shifted;
    generate
        for (i = 0; i < 2; i++) begin : gen_c_shift_l2
            assign c_l2_shifted[i] = {c_l2[i][11:0], 1'b0};
        end
    endgenerate

    // LEVEL 3: 1 13-bit 4:2 Compressor (4 inputs -> 2 outputs)
    logic [12:0] final_sum;
    logic [12:0] final_carry_raw;
    logic [13:0] cout_l3;

    assign cout_l3[0] = 1'b0;
    generate
        for (b = 0; b < 13; b++) begin : gen_csa_l3
            compressor_4to2 u_c42_l3 (
                .in0  (s_l2[0][b]),
                .in1  (c_l2_shifted[0][b]),
                .in2  (s_l2[1][b]),
                .in3  (c_l2_shifted[1][b]),
                .cin  (cout_l3[b]),
                .sum  (final_sum[b]),
                .carry(final_carry_raw[b]),
                .cout (cout_l3[b+1])
            );
        end
    endgenerate

    logic [12:0] final_carry;
    assign final_carry = {final_carry_raw[11:0], 1'b0};

    // FINAL ADDITION: Single 13-bit Carry-Propagate Adder
    logic signed [12:0] final_acc;
    assign final_acc = signed'(final_sum) + signed'(final_carry);

    assign dot_out = signed'({{3{final_acc[12]}}, final_acc});

endmodule
