`timescale 1ns / 1ps
module conv2d #(
    parameter BATCH_SIZE   = 1,
    parameter IN_CHANNELS  = 2,
    parameter OUT_CHANNELS = 1,
    parameter IN_HEIGHT    = 4,
    parameter IN_WIDTH     = 4,
    parameter KERNEL_SIZE  = 2,
    parameter STRIDE       = 2, // This parameter is correctly used below
    parameter PADDING      = 0, // This parameter is correctly used below
    parameter DATA_WIDTH   = 32
)(
    input clk,
    input rst,
    input start,
    output reg done,

    input  [BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH*DATA_WIDTH-1:0] input_tensor_flat,
    input  [OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_flat,
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] bias_flat,
    // Output tensor flat size calculation:
    // OUT_HEIGHT = (IN_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1
    // OUT_WIDTH  = (IN_WIDTH  + 2*PADDING - KERNEL_SIZE) / STRIDE + 1
    output reg [BATCH_SIZE*OUT_CHANNELS*((IN_HEIGHT + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1)*
                ((IN_WIDTH  + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1)*DATA_WIDTH-1:0] output_tensor_flat
);

    // Localparams for calculated output dimensions
    // These match the output port declaration.
    localparam OUT_HEIGHT = (IN_HEIGHT + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;
    localparam OUT_WIDTH  = (IN_WIDTH  + (2 * PADDING) - KERNEL_SIZE) / STRIDE + 1;

    // Local unpacked memories (reg is fine for synthesis as they are inferred as RAMs/registers)
    reg signed [DATA_WIDTH-1:0] input_tensor  [0:BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] weights       [0:OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE-1];
    reg signed [DATA_WIDTH-1:0] bias          [0:OUT_CHANNELS-1];
    reg signed [DATA_WIDTH-1:0] output_tensor [0:BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH-1];
    
    // Internal state for control
    localparam S_IDLE = 2'b00;
    localparam S_COMPUTE = 2'b01;
    localparam S_DONE = 2'b10;

    reg [1:0] current_state, next_state;

    // Declare loop iterators and temporary variables OUTSIDE the always @(posedge clk) block
    integer b, out_ch, in_ch, out_h, out_w, k_h, k_w;
    integer input_h, input_w;
    integer in_index, w_index, out_index;
    reg signed [DATA_WIDTH*2-1:0] acc_full_precision; // Accumulator, wider to prevent overflow
    reg signed [DATA_WIDTH-1:0] input_val_temp; // Temporary for input_val in the loop

    // FSM State Register
    always @(posedge clk or posedge rst) begin
        if (rst)
            current_state <= S_IDLE;
        else
            current_state <= next_state;
    end

    // FSM Next State Logic
    always @(*) begin
        next_state = current_state;
        case (current_state)
            S_IDLE: begin
                if (start)
                    next_state = S_COMPUTE;
            end
            S_COMPUTE: begin
                // This FSM transitions to S_DONE after one clock cycle of computation.
                // This implies a fully combinational convolution block,
                // or a block whose computation finishes in one clock cycle.
                // This is consistent with how it's used in residual_block.
                next_state = S_DONE;
            end
            S_DONE: begin
                if (!start) // Wait for external 'start' to go low before resetting
                    next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    // Done signal logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done <= 0;
        end else if (next_state == S_DONE && current_state == S_COMPUTE) begin
            // 'done' is asserted for one clock cycle when computation is complete.
            done <= 1;
        end else begin
            done <= 0;
        end
    end

    // Unpack input, weights, bias - these are combinational assignments
    // They happen continuously, providing current values to the computation logic.
    always @(*) begin
        for (integer i = 0; i < BATCH_SIZE*IN_CHANNELS*IN_HEIGHT*IN_WIDTH; i = i + 1)
            input_tensor[i] = input_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH];
        for (integer i = 0; i < OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE; i = i + 1)
            weights[i] = weights_flat[i*DATA_WIDTH +: DATA_WIDTH];
        for (integer i = 0; i < OUT_CHANNELS; i = i + 1)
            bias[i] = bias_flat[i*DATA_WIDTH +: DATA_WIDTH];
    end

    // Convolution Logic
    // This entire set of nested loops represents a single, large combinational block.
    // The 'always @(posedge clk or posedge rst)' makes it sequential, meaning the 
    // `output_tensor` registers are updated only once per clock cycle when in S_COMPUTE.
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (integer i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1)
                output_tensor[i] <= 0;
        end else if (current_state == S_COMPUTE) begin
            for (b = 0; b < BATCH_SIZE; b = b + 1) begin
                for (out_ch = 0; out_ch < OUT_CHANNELS; out_ch = out_ch + 1) begin
                    for (out_h = 0; out_h < OUT_HEIGHT; out_h = out_h + 1) begin
                        for (out_w = 0; out_w < OUT_WIDTH; out_w = out_w + 1) begin
                            acc_full_precision = bias[out_ch]; // Initialize with bias
                            for (in_ch = 0; in_ch < IN_CHANNELS; in_ch = in_ch + 1) begin
                                for (k_h = 0; k_h < KERNEL_SIZE; k_h = k_h + 1) begin
                                    for (k_w = 0; k_w < KERNEL_SIZE; k_w = k_w + 1) begin
                                        // Calculate input coordinates considering STRIDE and PADDING
                                        input_h = out_h * STRIDE + k_h - PADDING;
                                        input_w = out_w * STRIDE + k_w - PADDING;

                                        if (input_h >= 0 && input_h < IN_HEIGHT &&
                                            input_w >= 0 && input_w < IN_WIDTH) begin
                                            in_index = b*IN_CHANNELS*IN_HEIGHT*IN_WIDTH +
                                                       in_ch*IN_HEIGHT*IN_WIDTH +
                                                       input_h*IN_WIDTH + input_w;
                                            input_val_temp = input_tensor[in_index];
                                        end else begin
                                            input_val_temp = 0; // Zero padding
                                        end

                                        w_index = out_ch*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE +
                                                  in_ch*KERNEL_SIZE*KERNEL_SIZE +
                                                  k_h*KERNEL_SIZE + k_w;
                                        
                                        // Accumulate product
                                        acc_full_precision = acc_full_precision + (input_val_temp * weights[w_index]);
                                    end
                                end
                            end
                            out_index = b*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH +
                                        out_ch*OUT_HEIGHT*OUT_WIDTH +
                                        out_h*OUT_WIDTH + out_w;
                            // Assign the truncated result to the output tensor element
                            output_tensor[out_index] <= acc_full_precision[DATA_WIDTH-1:0];
                        end
                    end
                end
            end
        end
    end

    // Pack output tensor to flat (combinational assignment)
    // This 'always @(*)' block means output_tensor_flat will always reflect
    // the current contents of the 'output_tensor' array.
    always @(*) begin
        for (integer i = 0; i < BATCH_SIZE*OUT_CHANNELS*OUT_HEIGHT*OUT_WIDTH; i = i + 1)
            output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH] = output_tensor[i];
    end

endmodule