// This module is kept for completeness but its logic is absorbed into activation_wrapper
// for processing flattened tensors.
module leaky_relu #(
    parameter DATA_WIDTH = 16 // Example: 8 bits for Q1.7, 16 for Q1.15 etc.
)(
    input wire clk,
    input wire rst, // Changed from rst_n for consistency
    input wire signed [DATA_WIDTH-1:0] x_in,
    output reg signed [DATA_WIDTH-1:0] y_out
);
    // Alpha for Leaky ReLU is 1/128, which is a right shift by 7.
    // This assumes fixed-point format where shifting right by 7 correctly scales.
    // E.g., for Qm.n format, if alpha = 2^(-k), then x_in >>> k is suitable.
    // For Q1.7, alpha = 2^(-7) means right shift by 7, which will be mostly 0 or -1 (for negative values),
    // effectively truncating most precision. Be mindful of fixed-point format design.

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            y_out <= 0;
        end
        else begin
            if (x_in[DATA_WIDTH-1] == 1'b1) begin // Negative number
                y_out <= x_in >>> 7; // Arithmetic right shift by 7 (divide by 128)
            end
            else begin // Positive or zero
                y_out <= x_in;
            end
        end
    end

endmodule