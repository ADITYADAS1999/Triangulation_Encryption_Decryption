def xor_until_match(binary_input):
    result = binary_input[0]  # Initialize result with the first input bit
    outputs = [result]  # Store all output results
    while True:
        new_result = result[0]  # Start with the same first bit as the input
        for i in range(1, len(binary_input)):
            new_bit = str(int(result[i-1]) ^ int(binary_input[i]))  # XOR operation
            new_result += new_bit
        outputs.append(new_result)  # Store the new output
        if new_result == binary_input:
            break  # If the result matches the input, exit the loop
        result = new_result
    return outputs

# Take binary input from the user
binary_input = input("Enter a binary sequence: ")
operation_outputs = xor_until_match(binary_input)

# Print all operation outputs
print("Operation Outputs:")
for output in operation_outputs:
    print(output)
