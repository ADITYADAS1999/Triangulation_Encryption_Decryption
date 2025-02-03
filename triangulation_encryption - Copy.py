
def triangular_encrypt(bit_stream):
    # Initialize the triangular structure
    triangle = [list(bit_stream)]

    # Generate the triangle using XOR operation for each pair of consecutive bits
    for i in range(len(bit_stream) - 1):
        new_level = []
        for j in range(len(triangle[i]) - 1):
            # XOR operation between consecutive bits
            new_level.append(str(int(triangle[i][j]) ^ int(triangle[i][j + 1])))
        triangle.append(new_level)

    # This step should be adapted based on the specific extraction logic defined in the document
    encrypted_stream = ''.join(row[0] for row in triangle)  # Adjusted to match the example output

    return encrypted_stream

# Accept user input for the bit stream
bit_stream = input("Enter an 8-bit stream: ")

# Ensure the input is of correct length and format
if len(bit_stream) == 8 and all(bit in ['0', '1'] for bit in bit_stream):
    encrypted_stream = triangular_encrypt(bit_stream)
    print(f"Encrypted bit stream: {encrypted_stream}")
else:
    print("Invalid input. Please ensure you enter an 8-bit binary number.")



