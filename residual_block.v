`timescale 1ns / 1ps
module residual_block #(
    parameter DATA_WIDTH      = 8,
    parameter IN_CHANNELS     = 1,
    parameter IN_HEIGHT       = 4,
    parameter IN_WIDTH        = 4,
    parameter KERNEL_SIZE     = 2, // For conv2d (main path)
    parameter STRIDE          = 1, // Stride for conv2d (main path)
    parameter OUT_CHANNELS    = 1, // Final output channels for the block
    parameter USE_LEAKY_RELU  = 1,
    // Add parameters for projection shortcut control
    parameter USE_PROJECTION_SHORTCUT = 1 // 1 to use a 1x1 conv shortcut, 0 for identity (if dimensions match)
)(
    input clk,
    input rst,
    input start,
    output reg done,

    input  signed [DATA_WIDTH*IN_CHANNELS*IN_HEIGHT*IN_WIDTH-1:0] input_tensor,
    // Add inputs for weights and biases for all conv layers
    input  [OUT_CHANNELS*IN_CHANNELS*1*1*DATA_WIDTH-1:0] conv1_weights_flat,
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] conv1_bias_flat,
    
    input  [OUT_CHANNELS*IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] conv2_weights_flat,
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] conv2_bias_flat,
    
    input  [OUT_CHANNELS*OUT_CHANNELS*1*1*DATA_WIDTH-1:0] conv3_weights_flat,
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] conv3_bias_flat,

    // Weights and biases for the optional projection shortcut
    input  [OUT_CHANNELS*IN_CHANNELS*1*1*DATA_WIDTH-1:0] conv_shortcut_weights_flat, // For 1x1 conv on shortcut
    input  [OUT_CHANNELS*DATA_WIDTH-1:0] conv_shortcut_bias_flat,

    // Declared as 'reg' to allow procedural assignment inside 'always' block
    output reg signed [DATA_WIDTH*OUT_CHANNELS*( (IN_HEIGHT-KERNEL_SIZE)/STRIDE + 1 )*( (IN_WIDTH-KERNEL_SIZE)/STRIDE + 1 )-1:0] output_tensor
);

    // Calculate dimensions after conv2d (main path)
    localparam CONV2_OUT_HEIGHT = (IN_HEIGHT - KERNEL_SIZE) / STRIDE + 1;
    localparam CONV2_OUT_WIDTH  = (IN_WIDTH  - KERNEL_SIZE) / STRIDE + 1;
    localparam NUM_OUTPUT_ELEMENTS = OUT_CHANNELS * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH;

    // Wire declarations for intermediate outputs
    wire signed [DATA_WIDTH*OUT_CHANNELS*IN_HEIGHT*IN_WIDTH-1:0] conv1_out; // conv1x1 output channels are OUT_CHANNELS
    wire signed [DATA_WIDTH*OUT_CHANNELS*CONV2_OUT_HEIGHT*CONV2_OUT_WIDTH-1:0] conv2_out;
    wire signed [DATA_WIDTH*OUT_CHANNELS*CONV2_OUT_HEIGHT*CONV2_OUT_WIDTH-1:0] conv3_out;
    wire signed [DATA_WIDTH*NUM_OUTPUT_ELEMENTS-1:0] activated_out; // Output of main path before final add
    wire signed [DATA_WIDTH*NUM_OUTPUT_ELEMENTS-1:0] shortcut_out; // Output of shortcut path

    // Done signals from submodules
    wire conv1_done, conv2_done, conv3_done, act_done, conv_shortcut_done; // Renamed for clarity

    // State machine for residual block control
    localparam RB_IDLE = 4'd0; // Increased width to 4'd since we have 7 states plus optional shortcut
    localparam RB_CONV1 = 4'd1;
    localparam RB_CONV2 = 4'd2;
    localparam RB_CONV3 = 4'd3;
    localparam RB_ACT = 4'd4;
    localparam RB_SHORTCUT = 4'd5; // New state for shortcut convolution
    localparam RB_ADD = 4'd6;
    localparam RB_DONE = 4'd7;

    reg [3:0] current_rb_state, next_rb_state; // Increased width
    reg conv1_start_reg, conv2_start_reg, conv3_start_reg, act_start_reg, conv_shortcut_start_reg;

    // FSM State Register
    always @(posedge clk or posedge rst) begin
        if (rst)
            current_rb_state <= RB_IDLE;
        else
            current_rb_state <= next_rb_state;
    end

    // FSM Next State Logic
    always @(*) begin
        next_rb_state = current_rb_state;
        conv1_start_reg = 0;
        conv2_start_reg = 0;
        conv3_start_reg = 0;
        act_start_reg = 0;
        conv_shortcut_start_reg = 0;
        done = 0;

        case (current_rb_state)
            RB_IDLE: begin
                if (start) begin
                    conv1_start_reg = 1;
                    next_rb_state = RB_CONV1;
                end
            end
            RB_CONV1: begin
                if (conv1_done) begin
                    conv2_start_reg = 1;
                    next_rb_state = RB_CONV2;
                end else begin
                    conv1_start_reg = 1; // Keep start high until done
                end
            end
            RB_CONV2: begin
                if (conv2_done) begin
                    conv3_start_reg = 1;
                    next_rb_state = RB_CONV3;
                end else begin
                    conv2_start_reg = 1; // Keep start high until done
                end
            end
            RB_CONV3: begin
                if (conv3_done) begin
                    act_start_reg = 1;
                    next_rb_state = RB_ACT;
                end else begin
                    conv3_start_reg = 1; // Keep start high until done
                end
            end
            RB_ACT: begin
                if (act_done) begin
                    // After activation, check if a shortcut convolution is needed
                    if (USE_PROJECTION_SHORTCUT || (IN_CHANNELS != OUT_CHANNELS) || (STRIDE != 1) || (KERNEL_SIZE != 1) ) begin
                        // If any dimension or channel mismatch, or if forced, use projection shortcut
                        conv_shortcut_start_reg = 1;
                        next_rb_state = RB_SHORTCUT;
                    end else begin
                        // Direct add if dimensions/channels match (identity shortcut)
                        next_rb_state = RB_ADD;
                    end
                end else begin
                    act_start_reg = 1; // Keep start high until done
                end
            end
            RB_SHORTCUT: begin
                if (conv_shortcut_done) begin
                    next_rb_state = RB_ADD;
                end else begin
                    conv_shortcut_start_reg = 1; // Keep start high until done
                end
            end
            RB_ADD: begin
                // In this state, the combinatorial sum (comb_sum_out) is valid
                next_rb_state = RB_DONE; // Transition to DONE in the next cycle
            end
            RB_DONE: begin
                done = 1;
                if (!start) // Reset done when main 'start' goes low
                    next_rb_state = RB_IDLE;
            end
            default: next_rb_state = RB_IDLE;
        endcase
    end

    // conv1 (1x1 bottleneck/expansion)
    conv1x1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS), // Assuming conv1 changes channels from IN to OUT for block
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH)
    ) conv1 (
        .clk(clk),
        .rst(rst),
        .start(conv1_start_reg),
        .done(conv1_done),
        .input_tensor_flat(input_tensor),
        .weights_flat(conv1_weights_flat),
        .bias_flat(conv1_bias_flat),
        .output_tensor_flat(conv1_out)
    );

    // conv2 (main conv layer)
    conv2d #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(OUT_CHANNELS), // Input to conv2 comes from conv1_out (which has OUT_CHANNELS)
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE), // Pass STRIDE to conv2d
        .PADDING(0), // Assuming no padding for direct output dim calculation
        .IN_HEIGHT(IN_HEIGHT), // conv1_out has same spatial dimensions as input_tensor
        .IN_WIDTH(IN_WIDTH),
        .OUT_CHANNELS(OUT_CHANNELS) // conv2 outputs OUT_CHANNELS
    ) conv2 (
        .clk(clk),
        .rst(rst),
        .start(conv2_start_reg),
        .done(conv2_done),
        .input_tensor_flat(conv1_out),
        .weights_flat(conv2_weights_flat),
        .bias_flat(conv2_bias_flat),
        .output_tensor_flat(conv2_out)
    );

    // conv3 (1x1 projection back to desired output channels and matching dimensions of residual connection)
    conv1x1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(OUT_CHANNELS), // Input to conv3 comes from conv2_out (which has OUT_CHANNELS)
        .OUT_CHANNELS(OUT_CHANNELS), // Output channels are the block's OUT_CHANNELS
        .IN_HEIGHT(CONV2_OUT_HEIGHT),
        .IN_WIDTH(CONV2_OUT_WIDTH)
    ) conv3 (
        .clk(clk),
        .rst(rst),
        .start(conv3_start_reg),
        .done(conv3_done),
        .input_tensor_flat(conv2_out),
        .weights_flat(conv3_weights_flat),
        .bias_flat(conv3_bias_flat),
        .output_tensor_flat(conv3_out)
    );

    // Activation Wrapper for the main path (conv3_out)
    activation_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_ELEMENTS(NUM_OUTPUT_ELEMENTS), // Corrected to use calculated NUM_OUTPUT_ELEMENTS
        .USE_LEAKY_RELU(USE_LEAKY_RELU)
    ) act (
        .clk(clk),
        .rst(rst),
        .start(act_start_reg),
        .done(act_done), // Renamed from conv_act_done for clarity
        .input_tensor_flat(conv3_out),
        .output_tensor_flat(activated_out)
    );

    // Optional Projection Shortcut Convolution (1x1 with stride to match dimensions)
    generate
        // Check if a projection shortcut is truly needed
        if (USE_PROJECTION_SHORTCUT || (IN_CHANNELS != OUT_CHANNELS) || (STRIDE != 1) || (KERNEL_SIZE != 1) ) begin : shortcut_projection_needed
            conv1x1 #(
                .DATA_WIDTH(DATA_WIDTH),
                .IN_CHANNELS(IN_CHANNELS),
                .OUT_CHANNELS(OUT_CHANNELS),
                .IN_HEIGHT(IN_HEIGHT),
                .IN_WIDTH(IN_WIDTH),
                .STRIDE(STRIDE) // Pass stride to shortcut conv1x1
            ) conv_shortcut_inst (
                .clk(clk),
                .rst(rst),
                .start(conv_shortcut_start_reg),
                .done(conv_shortcut_done),
                .input_tensor_flat(input_tensor),
                .weights_flat(conv_shortcut_weights_flat),
                .bias_flat(conv_shortcut_bias_flat),
                .output_tensor_flat(shortcut_out)
            );
        end else begin : identity_shortcut_path
            // If no projection is needed (channels and spatial dims match, and stride is 1),
            // the shortcut is just the original input_tensor (identity mapping).
            // We need to ensure the `done` signal for this 'dummy' shortcut is handled.
            // For identity, we can assume it's "done" immediately.
            assign shortcut_out = input_tensor;
            assign conv_shortcut_done = 1'b1; // Combinatorially done
        end
    endgenerate

    // Wire to hold the combinatorial sum result
    wire signed [DATA_WIDTH*NUM_OUTPUT_ELEMENTS-1:0] combinatorial_sum_result;

    // This is the combinatorial addition logic, assigned using 'assign' for a wire
    // This will continuously reflect the sum of activated_out and shortcut_out
    assign combinatorial_sum_result = activated_out + shortcut_out;


    // Register the final output when the RB_ADD state is reached
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (integer i = 0; i < NUM_OUTPUT_ELEMENTS; i = i + 1) begin
                output_tensor[i*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
            end
        end else if (current_rb_state == RB_ADD) begin
            // Latch the combinatorial sum into the output_tensor registers
            output_tensor <= combinatorial_sum_result;
        end else if (current_rb_state == RB_DONE && !start) begin
            // Optional: clear output when done and external 'start' goes low
            // This ensures a clean state for the next computation.
            for (integer i = 0; i < NUM_OUTPUT_ELEMENTS; i = i + 1) begin
                output_tensor[i*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
            end
        end
        // Otherwise, output_tensor holds its previous value.
    end

    // Assertion for parameter consistency (optional but good practice)
    initial begin
        // If USE_PROJECTION_SHORTCUT is 0, ensure parameters allow direct add
        if (USE_PROJECTION_SHORTCUT == 0 && (IN_CHANNELS != OUT_CHANNELS || KERNEL_SIZE != 1 || STRIDE != 1) ) begin
            $display("WARNING: For identity shortcut (USE_PROJECTION_SHORTCUT=0), IN_CHANNELS must match OUT_CHANNELS, KERNEL_SIZE must be 1, and STRIDE must be 1. Current settings will result in a mismatch and likely incorrect residual addition.");
        end
    end

endmodule