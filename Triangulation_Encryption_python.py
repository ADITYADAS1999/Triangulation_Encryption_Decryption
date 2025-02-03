def xnor(a, b):
    return ~(a ^ b) & 1

def triangular_encrypt(bit_stream):
    blocks = [bit_stream]  # Adjusted to work on variable-length input
    encrypted_blocks = []
    for block in blocks:
        triangle = [block]
        while len(triangle[-1]) > 1:
            new_level = ''.join(str(xnor(int(triangle[-1][i]), int(triangle[-1][i+1]))) for i in range(len(triangle[-1])-1))
            triangle.append(new_level)

        # Forming the target block from the triangle (example: using MSBs)
        target_block = ''.join(level[0] for level in triangle)
        encrypted_blocks.append(target_block)

    # Combine all encrypted blocks
    encrypted_bit_stream = ''.join(encrypted_blocks)
    return encrypted_bit_stream

# Accept user input for the bit stream
bit_stream = input("Enter an 8-bit stream: ")  # Prompt the user to enter an 8-bit stream

# Ensure the input is of correct length and format
if len(bit_stream) == 8 and all(bit in ['0', '1'] for bit in bit_stream):
    encrypted_stream = triangular_encrypt(bit_stream)
    print(f"Encrypted bit stream: {encrypted_stream}")
else:
    print("Invalid input. Please ensure you enter an 8-bit binary number.")
