`timescale 1ns / 1ps

module activation_wrapper #(
    parameter DATA_WIDTH   = 8,
    parameter NUM_ELEMENTS = 16,
    parameter USE_LEAKY_RELU = 1 // 1 for Leaky ReLU, 0 for ReLU
)(
    input clk,
    input rst,
    input start,
    output reg done,

    input  signed [DATA_WIDTH*NUM_ELEMENTS-1:0] input_tensor_flat,
    output reg signed [DATA_WIDTH*NUM_ELEMENTS-1:0] output_tensor_flat
);

    // Internal unpacked array
    reg signed [DATA_WIDTH-1:0] input_elements [0:NUM_ELEMENTS-1];
    reg signed [DATA_WIDTH-1:0] output_elements [0:NUM_ELEMENTS-1];

    // Control FSM
    localparam S_IDLE    = 2'b00;
    localparam S_PROCESS = 2'b01;
    localparam S_DONE    = 2'b10;

    reg [1:0] current_state, next_state;

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
            S_IDLE:    if (start) next_state = S_PROCESS;
            S_PROCESS: next_state = S_DONE;
            S_DONE:    if (!start) next_state = S_IDLE;
        endcase
    end

    // Done signal logic
    always @(posedge clk or posedge rst) begin
        if (rst)
            done <= 0;
        else if (next_state == S_DONE && current_state == S_PROCESS)
            done <= 1;
        else
            done <= 0;
    end

    // Unpack input_tensor_flat
    integer i;
    always @(*) begin
        for (i = 0; i < NUM_ELEMENTS; i = i + 1)
            input_elements[i] = input_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH];
    end

    // Apply Activation (shared always block)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < NUM_ELEMENTS; i = i + 1)
                output_elements[i] <= 0;
        end else if (current_state == S_PROCESS) begin
            for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
                if (USE_LEAKY_RELU) begin
                    // Leaky ReLU: output = x if x >= 0 else x >> 7 (slope ~1/128)
                    if (input_elements[i][DATA_WIDTH-1] == 1'b1)
                        output_elements[i] <= input_elements[i] >>> 7;
                    else
                        output_elements[i] <= input_elements[i];
                end else begin
                    // ReLU: output = x if x >= 0 else 0
                    if (input_elements[i][DATA_WIDTH-1] == 1'b0)
                        output_elements[i] <= input_elements[i];
                    else
                        output_elements[i] <= {DATA_WIDTH{1'b0}};
                end
            end
        end
    end

    // Pack output_tensor_flat
    always @(*) begin
        for (i = 0; i < NUM_ELEMENTS; i = i + 1)
            output_tensor_flat[i*DATA_WIDTH +: DATA_WIDTH] = output_elements[i];
    end

endmodule
