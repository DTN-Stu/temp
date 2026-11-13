// =============================================================================
// Module: log_quantizer  (bản viết tay của bạn — chỉ sửa lỗi, giữ nguyên cấu trúc)
// -----------------------------------------------------------------------------
//   16 x FP16  ->  PACKET_WIDTH-bit packet = { scale_log8[7:0], fp4[15:0] }
//   Giải mã: value_i = sign_i * E2M1[mag_i] * 2^(scale_log8/8)
// =============================================================================
module log_quantizer #(
    parameter BLOCK_SIZE = 16,
    parameter PACKET_WIDTH = 72
) (
    input logic                          clk,
    input logic                          reset_n,

    input logic [BLOCK_SIZE-1:0][15:0]   fp16_in,
    input logic                          in_valid,
    output logic                         in_ready,

    output logic [PACKET_WIDTH-1:0]      packed_vec,
    output logic                         out_valid,
    input logic                          out_ready
);

    localparam signed [9:0]  SCALE_OFFSET = 9'sd140;

    // [FIX] 7 ngưỡng diff còn thiếu (trung điểm giữa các mức FP4, đơn vị 1/8 octave)
    localparam logic [7:0] TH_6P0 = 8'd2;
    localparam logic [7:0] TH_4P0 = 8'd6;
    localparam logic [7:0] TH_3P0 = 8'd10;
    localparam logic [7:0] TH_2P0 = 8'd14;
    localparam logic [7:0] TH_1P5 = 8'd18;
    localparam logic [7:0] TH_1P0 = 8'd24;
    localparam logic [7:0] TH_0P5 = 8'd36;   // biên 0.25 -> 8*log2(6/0.25) = 36.7

    // [FIX] thiếu return type; không được khai báo wire trong function
    function automatic logic [7:0] max2 (input logic [7:0] a, b);
        return (a > b) ? a : b;
    endfunction

    // Mã biên độ FP4 E2M1 từ số nấc thấp hơn đỉnh.
    function automatic logic [2:0] fp4_mag_of(input logic [7:0] diff);
        if      (diff <= TH_6P0) return 3'b111;  // 6.0
        else if (diff <= TH_4P0) return 3'b110;  // 4.0
        else if (diff <= TH_3P0) return 3'b101;  // 3.0
        else if (diff <= TH_2P0) return 3'b100;  // 2.0
        else if (diff <= TH_1P5) return 3'b011;  // 1.5
        else if (diff <= TH_1P0) return 3'b010;  // 1.0
        else if (diff <= TH_0P5) return 3'b001;  // 0.5
        else                     return 3'b000;  // 0.0
    endfunction


    logic [BLOCK_SIZE-1:0][4:0] E_fp16;
    logic [BLOCK_SIZE-1:0][9:0] M_fp16;
    logic [BLOCK_SIZE-1:0]      sign_fp16;   // [FIX] logoc -> logic

    logic [BLOCK_SIZE-1:0][7:0] mu;
    logic [7:0]                 mu_max;
    int                         slq_i_tree;
    int                         slq_i_seq;
    int                         slq_i_diff;
    int                         slq_i_out;

    generate
        // [FIX] generate loop cần genvar (không dùng int)
        for (genvar i=0; i<BLOCK_SIZE; i++) begin
            // [FIX] phải index [i] vào từng phần tử
            assign E_fp16[i]    = fp16_in[i][14:10];
            assign M_fp16[i]    = fp16_in[i][9:0];
            assign sign_fp16[i] = fp16_in[i][15];

            assign mu[i]        = (E_fp16[i] == 5'd0)  ? 8'd0   :
                                  (E_fp16[i] == 5'd31) ? 8'd247 :
                                                         {E_fp16[i], M_fp16[i][9:7]};
        end
    endgenerate


    // COMPARATION TREE
    logic [7:0][7:0] level1;
    logic [3:0][7:0] level2;
    logic [1:0][7:0] level3;

    always_comb begin
        for (slq_i_tree=0;slq_i_tree<8;slq_i_tree++) begin
             level1[slq_i_tree] = max2(mu[2*slq_i_tree], mu[2*slq_i_tree+1]);
        end
        for (slq_i_tree=0;slq_i_tree<4;slq_i_tree++) begin
             level2[slq_i_tree] = max2(level1[2*slq_i_tree], level1[2*slq_i_tree+1]);
        end
        for (slq_i_tree=0;slq_i_tree<2;slq_i_tree++) begin
             level3[slq_i_tree] = max2(level2[2*slq_i_tree], level2[2*slq_i_tree+1]);
        end
        mu_max = level3[0] > level3[1] ? level3[0]  : level3[1] ;
    end

    logic [BLOCK_SIZE-1:0][7:0] mu_d1;
    logic [7:0]                 mu_max_q1;
    logic [BLOCK_SIZE-1:0]      sign_fp16_q1;
    logic                       in_valid_q1;

    // Both pipeline stages share one stall: the output register is the only
    // elastic element, so the pipe can advance whenever it is free.  Exposing
    // this as `in_ready` turns the quantizer into a normal handshake stage
    // instead of an open pipe that silently drops packets under backpressure.
    wire pipe_advance = !out_valid || out_ready;
    assign in_ready   = pipe_advance;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            mu_max_q1 <= 'd0;
            sign_fp16_q1 <= 'd0;
            in_valid_q1 <= 1'b0;
            for (slq_i_seq=0; slq_i_seq<BLOCK_SIZE;slq_i_seq++) mu_d1[slq_i_seq] <= 'd0;
        end
        else if (pipe_advance) begin
            in_valid_q1 <= in_valid;
            if (in_valid) begin
                mu_max_q1    <= mu_max;
                sign_fp16_q1 <= sign_fp16;
                for (slq_i_seq=0; slq_i_seq<BLOCK_SIZE;slq_i_seq++) mu_d1[slq_i_seq] <= mu[slq_i_seq];
            end
        end
    end

    logic signed  [8:0] scale_raw ;
    logic signed  [7:0] scale_sat ;

    always_comb begin
        // [FIX] zero-check phải dùng mu_max_q1 (cùng tầng với phép trừ);
        //       tách if/else để nhánh zero không bị ghi đè
        if (mu_max_q1 == 8'd0) begin
            scale_sat = 8'sd0;
        end else begin
            // [FIX] zero-extend + $signed để tránh trộn signed/unsigned
            scale_raw = $signed({1'b0, mu_max_q1}) - SCALE_OFFSET;
            // [FIX] biên clamp đúng: 127 / -128 (8'sh80)
            if      (scale_raw >  9'sd127) scale_sat = 8'sd127;
            else if (scale_raw < -9'sd128) scale_sat = 8'sh80;
            else                           scale_sat = scale_raw[7:0];
        end
    end

    // DIFF CODE COMPUTING
    logic  [BLOCK_SIZE-1:0][7:0] diff_code;
    logic  [BLOCK_SIZE-1:0][3:0] fp4_code;
    logic  [BLOCK_SIZE-1:0][2:0] mag;

    always_comb begin
        for (slq_i_diff=0; slq_i_diff<BLOCK_SIZE; slq_i_diff++) begin
            
            diff_code[slq_i_diff] = mu_max_q1 - mu_d1[slq_i_diff];

            // [FIX] phần tử zero/quá nhỏ phải ra +0 (tránh -0); ngược lại ghép bit dấu
            mag[slq_i_diff] = (mu_d1[slq_i_diff] == 8'd0 || mu_max_q1 == 8'd0) ? 3'b000
                                                          : fp4_mag_of(diff_code[slq_i_diff]);
            fp4_code[slq_i_diff] = (mag[slq_i_diff] == 3'b000) ? 4'b0000 : {sign_fp16_q1[slq_i_diff], mag[slq_i_diff]};
        end
    end

    always_ff @(posedge  clk or negedge reset_n) begin
        if (!reset_n) begin
            out_valid <= 1'b0;
            packed_vec <= 'd0;
        end
        else if (pipe_advance) begin
            out_valid <= in_valid_q1;
            // [FIX] chỉ đóng gói khi có dữ liệu hợp lệ
            if (in_valid_q1) begin
                for (slq_i_out=0; slq_i_out<BLOCK_SIZE; slq_i_out++) begin
                    // [FIX] mỗi mã 4-bit, không phải 16-bit
                    packed_vec[4*slq_i_out +: 4] <= fp4_code[slq_i_out];
                end
                packed_vec[PACKET_WIDTH-1 -: 8] <= scale_sat;
            end
        end
    end

endmodule
