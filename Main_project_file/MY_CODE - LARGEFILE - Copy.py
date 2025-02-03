from tkinter import *                                                                                                                                                                                                                                                                                                                                                           
import tkinter.messagebox as messagebox
import speech_recognition as sr
import threading
import time
import random
import time
from tkinter import END, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, Button
import tkinter as tk



root = Tk()
root.title("Triangulation Encryption & Decryption Data")
root.geometry("1920x1000+0+0")

# Global variables to store encryption and decryption time
total_encryption_time = 0.0
total_decryption_time = 0.0
stored_key = None  # Global variable to store the encryption key
final_cipher_text = ""

global chi_square_value, degrees_of_freedom
chi_square_value = 0
degrees_of_freedom = 1

#======================================================================================================


start_time = time.time()

def encrypt_decrypt(text, key):
    encrypted = ''.join(chr(ord(x) ^ key) for x in text)
    return encrypted




def divide_into_blocks(bit_stream):
    blocks = []
    block_size = 8  # Fixed block size to 8 bits
    while bit_stream:
        if len(bit_stream) < block_size:
            block_size = len(bit_stream)
        blocks.append(bit_stream[:block_size])
        bit_stream = bit_stream[block_size:]
    return blocks




from collections import Counter  # Make sure to import this for the Counter class

# Function to count bits in a binary stream
def count_bits(binary_text):
    return Counter(binary_text)

# Function to calculate chi-square value
from fractions import Fraction
from collections import Counter

from fractions import Fraction
from collections import Counter

from fractions import Fraction


"""


def calculate_chi_square(original_counts, encrypted_counts):
    # Calculate total counts for original and encrypted data
    total_original = sum(original_counts.values())
    total_encrypted = sum(encrypted_counts.values())

    # Convert counts to Fraction for precision
    original_counts = {k: Fraction(v) for k, v in original_counts.items()}
    encrypted_counts = {k: Fraction(v) for k, v in encrypted_counts.items()}

    # Calculate expected counts for '0' and '1'
    total_count = total_original + total_encrypted
    if total_count == 0:
        return Fraction(0), 1  # Return 0 chi-square if no counts are available

    # Calculate expected counts (average of original and encrypted counts)
    expected_0 = (original_counts.get('0', Fraction(0)) + encrypted_counts.get('0', Fraction(0))) / 2
    expected_1 = (original_counts.get('1', Fraction(0)) + encrypted_counts.get('1', Fraction(0))) / 2

    # Initialize chi-square numerator as Fraction to ensure precision
    chi_square_numerator = Fraction(0)

    # Chi-square formula for '0's
    if expected_0 > 0:
        chi_square_numerator += ((original_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0
        chi_square_numerator += ((encrypted_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0

    # Chi-square formula for '1's
    if expected_1 > 0:
        chi_square_numerator += ((original_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1
        chi_square_numerator += ((encrypted_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1

    # Return chi-square value and degree of freedom (1 for binary sequence)
    return chi_square_numerator, 1

    degree_of_freedom = len(sequences[-1]) - 1  # For each block, degree of freedom is the length of the final sequence minus one




# Function to calculate frequency distribution of characters in the text
def calculate_frequency_distribution(text):
    frequency = Counter(text)
    total_characters = sum(frequency.values())
    frequency_distribution = {char: count / total_characters for char, count in frequency.items()}
    return frequency_distribution


# Define the degree of freedom formula
def calculate_degree_of_freedom(final_sequence_length):
    return final_sequence_length - 1

"""









# Generate a session key with 8 segments, each 16 bits long
def generate_session_key():
    segments = []
    for _ in range(8):
        segment = random.randint(0, 2**16 - 1)  # 16-bit segment
        segments.append(segment)
    return segments



# Convert content to binary chunks based on shuffled bit lengths
def content_to_binary_chunks(content, shuffled_lengths):
    binary_chunks = []
    index = 0
    for length in shuffled_lengths:
        chunk = content[index:index + (length // 8)]
        if chunk:  # Ensure chunk is not empty
            binary_chunk = ''.join(format(ord(c), '08b') for c in chunk)
            binary_chunks.append(binary_chunk)
        index += (length // 8)
    return binary_chunks



"""

from collections import Counter  # Make sure to import this for the Counter class

# Function to count bits in a binary stream
def count_bits(binary_text):
    return Counter(binary_text)

# Function to calculate chi-square value
from fractions import Fraction
from collections import Counter

from fractions import Fraction
from collections import Counter

def calculate_chi_square(original_counts, encrypted_counts):
    # Calculate total counts
    total_original = sum(original_counts.values())
    total_encrypted = sum(encrypted_counts.values())

    # Convert counts to Fraction to ensure precision
    original_counts = {k: Fraction(v) for k, v in original_counts.items()}
    encrypted_counts = {k: Fraction(v) for k, v in encrypted_counts.items()}

    # Expected counts
    total_count = total_original + total_encrypted
    if total_count == 0:
        return Fraction(0), 1  # Return 0 chi-square if no counts are available

    expected_0 = (original_counts.get('0', Fraction(0)) + encrypted_counts.get('0', Fraction(0))) / 2
    expected_1 = (original_counts.get('1', Fraction(0)) + encrypted_counts.get('1', Fraction(0))) / 2

    # Initialize the numerator
    chi_square_numerator = Fraction(0)

    # Chi-square calculation
    if expected_0 > 0:
        chi_square_numerator += ((original_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0
        chi_square_numerator += ((encrypted_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0

    if expected_1 > 0:
        chi_square_numerator += ((original_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1
        chi_square_numerator += ((encrypted_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1

    return chi_square_numerator, 1  # Degree of freedom for binary sequence is 1
  # Degree of freedom for binary sequence is 1
 # Degree of freedom for binary sequence is 1


# Function to calculate frequency distribution of characters in the text
def calculate_frequency_distribution(text):
    frequency = Counter(text)
    total_characters = sum(frequency.values())
    frequency_distribution = {char: count / total_characters for char, count in frequency.items()}
    return frequency_distribution





def encrypt():
    global total_encryption_time, stored_key, final_cipher_text, xor_results
    try:
        start_time = time.time()

        # Step 1: Convert plaintext to binary (8-bit binary input)
        plaintext = plaintext_text.get("1.0", END).strip()
        binary_text = text_to_binary(plaintext)
        triangulation_entry.delete("1.0", END)
        triangulation_entry.insert(END, "Binary Input String:\n" + binary_text + "\n")

        # Step 2: Divide the binary input string into 8-bit blocks
        blocks = divide_into_blocks(binary_text)
        triangulation_entry.insert(END, "\n8-bit Blocks:\n" + " / ".join(blocks) + "\n")

        # Step 3: Generate and display the 128-bit session key
        session_key = generate_session_key()
        key_binary = ''.join(f'{segment:016b}' for segment in session_key)
        stored_key = int(key_binary, 2)

        triangulation_entry.insert(END, "\nSession Key (Decimal):\n" + str(stored_key) + "\n")
        triangulation_entry.insert(END, "\nSession Key (Binary):\n" + key_binary + "\n")

        # Step 4: Apply XOR operation between input data and session key
        xor_result = ''
        key_segment_index = 0
        xor_results = []  # Store XOR results for comparison during decryption
        for block in blocks:
            key_segment = key_binary[key_segment_index:key_segment_index + 8]
            key_segment_index = (key_segment_index + 8) % len(key_binary)
            xor_block = ''.join(str(int(block[i]) ^ int(key_segment[i])) for i in range(8))
            xor_results.append(xor_block)
            xor_result += xor_block

        triangulation_entry.insert(END, "\nDATA XOR KEY:\n" + xor_result + "\n")

        # Step 5: Apply triangulation encryption to the XOR result
        final_cipher_text, triangle_output, options_used = triangular_encrypt_block(xor_result)
        triangulation_entry.insert(END, "\nTarget Blocks After Triangulation Encryption:\n" + "\n".join(triangle_output) + "\n")

        # Step 6: Generate the final cipher text and display it
        final_cipher_text = ''.join([ascii_to_cipher(binary_to_ascii_numeric(block)) for block in triangle_output])
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")

        # Step 7: Calculate and display Chi-Square value, degree of freedom, and frequency distributions
        original_counts = count_bits(binary_text)
        encrypted_counts = count_bits(xor_result)

        # Chi-Square Calculation
        chi_square_value, degrees_of_freedom = calculate_chi_square(original_counts, encrypted_counts)
        triangulation_entry.insert(END, f"\nChi-Square Value: {float(chi_square_value):.5f}\n")
        triangulation_entry.insert(END, f"Degree of Freedom: {degrees_of_freedom}\n")

        # Frequency Distribution Calculation for original and encrypted text
        original_frequency = calculate_frequency_distribution(plaintext)
        encrypted_frequency = calculate_frequency_distribution(final_cipher_text)

        triangulation_entry.insert(END, "\nFrequency Distribution of Original Text:\n")
        for char, freq in original_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")

        triangulation_entry.insert(END, "\nFrequency Distribution of Encrypted Text:\n")
        for char, freq in encrypted_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")

    except Exception as e:
        messagebox.showerror('Error', str(e))

    # Update encryption time
    encryption_duration = time.time() - start_time
    total_encryption_time += encryption_duration
    update_time_labels()

"""

"""

total_encryption_time = 0.0

def encrypt():
    global total_encryption_time, stored_key, final_cipher_text, xor_results
    try:
        start_time = time.time()

        # Step 1: Convert plaintext to binary (8-bit binary input)
        plaintext = plaintext_text.get("1.0", END).strip()
        binary_text = text_to_binary(plaintext)
        triangulation_entry.delete("1.0", END)
        triangulation_entry.insert(END, "Binary Input String:\n" + binary_text + "\n")

        # Step 2: Divide the binary input string into 8-bit blocks
        blocks = divide_into_blocks(binary_text)
        triangulation_entry.insert(END, "\n8-bit Blocks:\n" + " / ".join(blocks) + "\n")

        # Step 3: Generate and display the 128-bit session key
        session_key = generate_session_key()
        key_binary = ''.join(f'{segment:016b}' for segment in session_key)
        stored_key = int(key_binary, 2)

        triangulation_entry.insert(END, "\nSession Key (Decimal):\n" + str(stored_key) + "\n")
        triangulation_entry.insert(END, "\nSession Key (Binary):\n" + key_binary + "\n")

        # Step 4: Apply XOR operation between input data and session key
        xor_result = ''
        key_segment_index = 0
        xor_results = []  # Store XOR results for comparison during decryption
        for block in blocks:
            key_segment = key_binary[key_segment_index:key_segment_index+8]
            key_segment_index = (key_segment_index + 8) % len(key_binary)
            xor_block = ''.join(str(int(block[i]) ^ int(key_segment[i])) for i in range(8))
            xor_results.append(xor_block)
            xor_result += xor_block

        triangulation_entry.insert(END, "\nDATA XOR KEY:\n" + xor_result + "\n")

        # Step 5: Apply triangulation encryption to the XOR result
        final_cipher_text, triangle_output, options_used = triangular_encrypt_block(xor_result)
        triangulation_entry.insert(END, "\nTarget Blocks After Triangulation Encryption:\n" + "\n".join(triangle_output) + "\n")

        # Step 6: Generate the final cipher text and display it
        final_cipher_text = ''.join([ascii_to_cipher(binary_to_ascii_numeric(block)) for block in triangle_output])
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")
        

        

    except Exception as e:
        messagebox.showerror('Error', str(e))


        # Update encryption time
        encryption_duration = time.time() - start_time
        total_encryption_time += encryption_duration
        update_time_labels()

"""








"""
from collections import Counter
from fractions import Fraction
import random
import time
import threading
import speech_recognition as sr
from tkinter import END, messagebox

# Function to count bits in a binary stream
def count_bits(binary_text):
    return Counter(binary_text)

# Function to calculate chi-square value
def calculate_chi_square(original_counts, encrypted_counts):
    total_original = sum(original_counts.values())
    total_encrypted = sum(encrypted_counts.values())

    # Convert counts to Fraction for precision
    original_counts = {k: Fraction(v) for k, v in original_counts.items()}
    encrypted_counts = {k: Fraction(v) for k, v in encrypted_counts.items()}

    total_count = total_original + total_encrypted
    if total_count == 0:
        return Fraction(0), 1  # Return 0 chi-square if no counts are available

    expected_0 = (original_counts.get('0', Fraction(0)) + encrypted_counts.get('0', Fraction(0))) / 2
    expected_1 = (original_counts.get('1', Fraction(0)) + encrypted_counts.get('1', Fraction(0))) / 2

    chi_square_numerator = Fraction(0)

    if expected_0 > 0:
        chi_square_numerator += ((original_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0
        chi_square_numerator += ((encrypted_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0

    if expected_1 > 0:
        chi_square_numerator += ((original_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1
        chi_square_numerator += ((encrypted_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1

    return chi_square_numerator, 1  # Degrees of freedom for binary sequence is 1

# Function to calculate frequency distribution of characters in the text
def calculate_frequency_distribution(text):
    frequency = Counter(text)
    total_characters = sum(frequency.values())
    frequency_distribution = {char: count / total_characters for char, count in frequency.items()}
    return frequency_distribution

# Function to convert text to binary
def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

# Function to convert binary to text
def binary_to_text(binary_str):
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))

# Function to generate a random 128-bit session key
def generate_session_key():
    return [random.randint(0, 65535) for _ in range(8)]  # 8 segments, each 16 bits

# Function to apply triangulation to a bit stream
def apply_triangulation_to_block(bit_stream, option, block_index):
    triangle = [list(bit_stream)]
    local_output = [f"Block-{block_index + 1} Iteration-1 {' '.join(list(bit_stream))}"]
    
    for iteration_index in range(len(bit_stream) - 1):
        new_iteration = []
        for j in range(len(triangle[iteration_index]) - 1):
            new_bit = str(int(triangle[iteration_index][j]) ^ int(triangle[iteration_index][j + 1]))
            new_iteration.append(new_bit)
        triangle.append(new_iteration)
        local_output.append(f"Iteration-{iteration_index + 2} {' '.join(new_iteration)}")

    # Determine encrypted stream based on the option
    if option == '001':
        encrypted_stream = ''.join(row[0] for row in triangle)
        arrow = "↓ (MSB source to last | 001)"
    elif option == '010':
        encrypted_stream = ''.join(row[0] for row in reversed(triangle))
        arrow = "↑ (MSB last to source | 010)"
    elif option == '011':
        encrypted_stream = ''.join(row[-1] for row in triangle)
        arrow = "↓ (LSB source to last | 011)"
    else:  # '100'
        encrypted_stream = ''.join(row[-1] for row in reversed(triangle))
        arrow = "↑ (LSB last to source | 100)"
    
    local_output[0] += " " + arrow
    msb = encrypted_stream[0] if encrypted_stream else "N/A"
    lsb = encrypted_stream[-1] if encrypted_stream else "N/A"
    
    local_output.append(f"\nGenerated Target Block: {encrypted_stream} \n")
    local_output.append(f"MSB: {msb} | LSB: {lsb}")

    return '\n'.join(local_output[:-1]), encrypted_stream, msb, lsb, encrypted_stream

# Function to encrypt plaintext
def encrypt():
    global total_encryption_time, stored_key, final_cipher_text, xor_results
    try:
        start_time = time.time()
        plaintext = plaintext_text.get("1.0", END).strip()
        binary_text = text_to_binary(plaintext)
        
        # Divide the binary input into 8-bit blocks
        blocks = divide_into_blocks(binary_text)
        
        # Generate and display a random 128-bit session key
        session_key = generate_session_key()
        key_binary = ''.join(f'{segment:016b}' for segment in session_key)
        stored_key = int(key_binary, 2)

        # XOR operation between input data and session key
        xor_result = ''
        key_segment_index = 0
        xor_results = []  
        for block in blocks:
            key_segment = key_binary[key_segment_index:key_segment_index + 8]
            key_segment_index = (key_segment_index + 8) % len(key_binary)
            xor_block = ''.join(str(int(block[i]) ^ int(key_segment[i])) for i in range(8))
            xor_results.append(xor_block)
            xor_result += xor_block

        # Triangulation encryption
        final_cipher_text, triangle_output, options_used = triangular_encrypt_block(xor_result)
        
        # Chi-Square Calculation
        original_counts = count_bits(binary_text)
        encrypted_counts = count_bits(xor_result)
        chi_square_value, degrees_of_freedom = calculate_chi_square(original_counts, encrypted_counts)
        
        # Frequency Distribution Calculation
        original_frequency = calculate_frequency_distribution(plaintext)
        encrypted_frequency = calculate_frequency_distribution(final_cipher_text)

        # Display results in UI
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")
        triangulation_entry.insert(END, f"\nChi-Square Value: {float(chi_square_value):.5f}\n")
        triangulation_entry.insert(END, f"Degree of Freedom: {degrees_of_freedom}\n")
        triangulation_entry.insert(END, "\nFrequency Distribution of Original Text:\n")
        for char, freq in original_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")
        triangulation_entry.insert(END, "\nFrequency Distribution of Encrypted Text:\n")
        for char, freq in encrypted_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")

    except Exception as e:
        messagebox.showerror('Error', str(e))

    encryption_duration = time.time() - start_time
    total_encryption_time += encryption_duration
    update_time_labels()


"""

total_encryption_time = 0.0


from collections import Counter  # Make sure to import this for the Counter class

# Function to count bits in a binary stream
def count_bits(binary_text):
    return Counter(binary_text)

# Function to calculate chi-square value
from fractions import Fraction
from collections import Counter

from fractions import Fraction
from collections import Counter

def calculate_chi_square(original_counts, encrypted_counts):
    # Calculate total counts
    total_original = sum(original_counts.values())
    total_encrypted = sum(encrypted_counts.values())

    # Convert counts to Fraction to ensure precision
    original_counts = {k: Fraction(v) for k, v in original_counts.items()}
    encrypted_counts = {k: Fraction(v) for k, v in encrypted_counts.items()}

    # Expected counts
    total_count = total_original + total_encrypted
    if total_count == 0:
        return Fraction(0), 1  # Return 0 chi-square if no counts are available

    expected_0 = (original_counts.get('0', Fraction(0)) + encrypted_counts.get('0', Fraction(0))) / 2
    expected_1 = (original_counts.get('1', Fraction(0)) + encrypted_counts.get('1', Fraction(0))) / 2

    # Initialize the numerator
    chi_square_numerator = Fraction(0)

    # Chi-square calculation
    if expected_0 > 0:
        chi_square_numerator += ((original_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0
        chi_square_numerator += ((encrypted_counts.get('0', Fraction(0)) - expected_0) ** 2) / expected_0

    if expected_1 > 0:
        chi_square_numerator += ((original_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1
        chi_square_numerator += ((encrypted_counts.get('1', Fraction(0)) - expected_1) ** 2) / expected_1

    return chi_square_numerator, 1  # Degree of freedom for binary sequence is 1


    degree_of_freedom = len(sequences[-1]) - 1  # For each block, degree of freedom is the length of the final sequence minus one
  # Degree of freedom for binary sequence is 1
 # Degree of freedom for binary sequence is 1


# Define the degree of freedom formula
def calculate_degree_of_freedom(final_sequence_length):
    return final_sequence_length - 1


# Function to calculate frequency distribution of characters in the text
def calculate_frequency_distribution(text):
    frequency = Counter(text)
    total_characters = sum(frequency.values())
    frequency_distribution = {char: count / total_characters for char, count in frequency.items()}
    return frequency_distribution


"""
def encrypt():
    global total_encryption_time, stored_key, final_cipher_text, xor_results
    try:
        start_time = time.time()

        # Step 1: Convert plaintext to binary (8-bit binary input)
        plaintext = plaintext_text.get("1.0", END).strip()
        binary_text = text_to_binary(plaintext)
        triangulation_entry.delete("1.0", END)
        triangulation_entry.insert(END, "Binary Input String:\n" + binary_text + "\n")

        # Step 2: Divide the binary input string into 8-bit blocks
        blocks = divide_into_blocks(binary_text)
        triangulation_entry.insert(END, "\n8-bit Blocks:\n" + " / ".join(blocks) + "\n")

        # Step 3: Generate and display the 128-bit session key
        session_key = generate_session_key()
        key_binary = ''.join(f'{segment:016b}' for segment in session_key)
        stored_key = int(key_binary, 2)

        triangulation_entry.insert(END, "\nSession Key (Decimal):\n" + str(stored_key) + "\n")
        triangulation_entry.insert(END, "\nSession Key (Binary):\n" + key_binary + "\n")

        # Step 4: Apply XOR operation between input data and session key
        xor_result = ''
        key_segment_index = 0
        xor_results = []  # Store XOR results for comparison during decryption
        for block in blocks:
            key_segment = key_binary[key_segment_index:key_segment_index + 8]
            key_segment_index = (key_segment_index + 8) % len(key_binary)
            xor_block = ''.join(str(int(block[i]) ^ int(key_segment[i])) for i in range(8))
            xor_results.append(xor_block)
            xor_result += xor_block

        triangulation_entry.insert(END, "\nDATA XOR KEY:\n" + xor_result + "\n")

        # Step 5: Apply triangulation encryption to the XOR result
        final_cipher_text, triangle_output, options_used = triangular_encrypt_block(xor_result)
        triangulation_entry.insert(END, "\nTarget Blocks After Triangulation Encryption:\n" + "\n".join(triangle_output) + "\n")

        # Step 6: Generate the final cipher text and display it
        final_cipher_text = ''.join([ascii_to_cipher(binary_to_ascii_numeric(block)) for block in triangle_output])
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")

        # Step 7: Calculate and display Chi-Square value, degree of freedom, and frequency distributions
        original_counts = count_bits(binary_text)
        encrypted_counts = count_bits(xor_result)


        

        # Chi-Square Calculation
        chi_square_value, degrees_of_freedom = calculate_chi_square(original_counts, encrypted_counts)
        triangulation_entry.insert(END, f"\nChi-Square Value: {float(chi_square_value):.5f}\n")
        triangulation_entry.insert(END, f"Degree of Freedom: {degrees_of_freedom}\n")

        # Frequency Distribution Calculation for original and encrypted text
        original_frequency = calculate_frequency_distribution(plaintext)
        encrypted_frequency = calculate_frequency_distribution(final_cipher_text)

        triangulation_entry.insert(END, "\nFrequency Distribution of Original Text:\n")
        for char, freq in original_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")

        triangulation_entry.insert(END, "\nFrequency Distribution of Encrypted Text:\n")
        for char, freq in encrypted_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")
            

    except Exception as e:
        messagebox.showerror('Error', str(e))

    # Update encryption time
    encryption_duration = time.time() - start_time
    total_encryption_time += encryption_duration
    update_time_labels()
"""


"""
total_encryption_time = 0.0



def encrypt():
    global total_encryption_time, stored_key, final_cipher_text, xor_results
    start_time = time.time()
    try:
        #start_time = time.time()
        plaintext = plaintext_text.get("1.0", END).strip()
        binary_text = text_to_binary(plaintext)
        
        triangulation_entry.delete("1.0", END)
        triangulation_entry.insert(END, "Binary Input String:\n" + binary_text + "\n")

        # Divide the binary input into 8-bit blocks
        blocks = divide_into_blocks(binary_text)
        triangulation_entry.insert(END, "\n8-bit Blocks:\n" + " / ".join(blocks) + "\n")

        # Generate session key
        session_key = generate_session_key()
        key_binary = ''.join(f'{segment:016b}' for segment in session_key)
        stored_key = int(key_binary, 2)
        triangulation_entry.insert(END, "\nSession Key (Decimal):\n" + str(stored_key) + "\n")
        triangulation_entry.insert(END, "\nSession Key (Binary):\n" + key_binary + "\n")

        # XOR operation
        xor_result = ''
        key_segment_index = 0
        xor_results = []  # Store XOR results for comparison during decryption
        for block in blocks:
            key_segment = key_binary[key_segment_index:key_segment_index + 8]
            key_segment_index = (key_segment_index + 8) % len(key_binary)
            xor_block = ''.join(str(int(block[i]) ^ int(key_segment[i])) for i in range(8))
            xor_results.append(xor_block)
            xor_result += xor_block

        triangulation_entry.insert(END, "\nDATA XOR KEY:\n" + xor_result + "\n")

        # Triangulation encryption
        final_cipher_text, triangle_output, options_used = triangular_encrypt_block(xor_result)
        triangulation_entry.insert(END, "\nTarget Blocks After Triangulation Encryption:\n" + "\n".join(triangle_output) + "\n")

        # Display final cipher text
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")

        # Chi-Square and frequency distribution calculations
        original_counts = count_bits(binary_text)
        encrypted_counts = count_bits(xor_result)
        chi_square_value, degrees_of_freedom = calculate_chi_square(original_counts, encrypted_counts)
        
        triangulation_entry.insert(END, f"\nChi-Square Value: {float(chi_square_value):.5f}\n")
        triangulation_entry.insert(END, f"Degree of Freedom: {degrees_of_freedom}\n")

    # Update encryption time
    encryption_duration = time.time() - start_time
    total_encryption_time += encryption_duration
    update_time_labels()
        original_frequency = calculate_frequency_distribution(plaintext)
        encrypted_frequency = calculate_frequency_distribution(final_cipher_text)

        triangulation_entry.insert(END, "\nFrequency Distribution of Original Text:\n")
        for char, freq in original_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")

        triangulation_entry.insert(END, "\nFrequency Distribution of Encrypted Text:\n")
        for char, freq in encrypted_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")


    except Exception as e:
        messagebox.showerror('Error', str(e))

    # Update encryption time
    encryption_duration = time.time() - start_time
    total_encryption_time += encryption_duration
    update_time_labels()

"""


def encrypt():
    global total_encryption_time, stored_key, final_cipher_text, xor_results, chi_square_value, degrees_of_freedom
    start_time = time.time()
    try:
        plaintext = plaintext_text.get("1.0", END).strip()
        binary_text = text_to_binary(plaintext)
        triangulation_entry.delete("1.0", END)
        triangulation_entry.insert(END, "Binary Input String:\n" + binary_text + "\n")

        # Divide binary input into 8-bit blocks
        blocks = divide_into_blocks(binary_text)
        triangulation_entry.insert(END, "\n8-bit Blocks:\n" + " / ".join(blocks) + "\n")

        # Generate session key
        session_key = generate_session_key()
        key_binary = ''.join(f'{segment:016b}' for segment in session_key)
        stored_key = int(key_binary, 2)
        triangulation_entry.insert(END, "\nSession Key (Decimal):\n" + str(stored_key) + "\n")
        triangulation_entry.insert(END, "\nSession Key (Binary):\n" + key_binary + "\n")

        # XOR operation
        xor_result = ''
        key_segment_index = 0
        xor_results = []
        for block in blocks:
            key_segment = key_binary[key_segment_index:key_segment_index + 8]
            if len(key_segment) < 8:
                key_segment = key_segment.ljust(8, '0')
            key_segment_index = (key_segment_index + 8) % len(key_binary)
            xor_block = ''.join(str(int(block[i]) ^ int(key_segment[i])) for i in range(len(block)))
            xor_results.append(xor_block)
            xor_result += xor_block

        triangulation_entry.insert(END, "\nDATA XOR KEY:\n" + xor_result + "\n")

        # Triangulation encryption
        final_cipher_text, triangle_output, options_used = triangular_encrypt_block(xor_result)
        triangulation_entry.insert(END, "\nTarget Blocks After Triangulation Encryption:\n" + "\n".join(triangle_output) + "\n")
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")

        # Chi-Square and frequency distribution calculations
        original_counts = count_bits(binary_text)
        encrypted_counts = count_bits(xor_result)

        # Debugging output to check counts
        print(f"Original Counts: {original_counts}")
        print(f"Encrypted Counts: {encrypted_counts}")

        chi_square_value, degrees_of_freedom = calculate_chi_square(original_counts, encrypted_counts)
        triangulation_entry.insert(END, f"\nChi-Square Value: {float(chi_square_value):.6f}\n")
        triangulation_entry.insert(END, f"Degree of Freedom: {degrees_of_freedom}\n")

        original_frequency = calculate_frequency_distribution(plaintext)
        encrypted_frequency = calculate_frequency_distribution(final_cipher_text)

        triangulation_entry.insert(END, "\nFrequency Distribution of Original Text:\n")
        for char, freq in original_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")

        triangulation_entry.insert(END, "\nFrequency Distribution of Encrypted Text:\n")
        for char, freq in encrypted_frequency.items():
            triangulation_entry.insert(END, f"Character: {char}, Frequency: {freq:.4f}\n")
        triangulation_entry.insert(END, "\nSession Key (Decimal):\n" + str(stored_key) + "\n")
        triangulation_entry.insert(END, f"\nChi-Square Value: {float(chi_square_value):.6f}\n")
        triangulation_entry.insert(END, "\nTHE ENCRYPTED CIPHER TEXT:\n" + final_cipher_text + "\n")

    except Exception as e:
        print(f"Error encountered: {e}")
        messagebox.showerror('Error', str(e))

    # Update encryption time
    encryption_duration = time.time() - start_time
    total_encryption_time += encryption_duration
    update_time_labels()


"""
def decrypt():
    global total_decryption_time, stored_key, final_cipher_text, used_options, original_source_blocks, xor_results
    try:
        if stored_key is None:
            messagebox.showerror('Authentication Error', 'The key has been destroyed. Decryption is not possible.')
            return

        start_time = time.time()

        # Convert the stored session key (decimal) to binary
        key_binary = f'{stored_key:0128b}'  # Convert stored key to a 128-bit binary string
        print(f"Session Key (128-bit Binary): {key_binary}")

        # Split the 128-bit session key into 8-bit segments
        key_segments = [key_binary[i:i+8] for i in range(0, 128, 8)]
        print(f"Session Key Segments (8-bit): {key_segments}")

        # Use the stored encrypted cipher text directly
        encrypted_text = final_cipher_text

        # Convert the encrypted text to binary stream
        encrypted_binary = text_to_binary(encrypted_text)

        # Divide the encrypted binary stream into 8-bit blocks
        blocks = divide_into_blocks(encrypted_binary)

        # Debugging output
        print(f"Encrypted text: {encrypted_text}")
        print(f"Encrypted binary: {encrypted_binary}")
        print(f"Blocks: {blocks}")

        # Create a string for binary blocks separated by '/'
        binary_blocks_str = '/'.join(blocks)

        # Display the heading and encrypted cipher text
        triangulation_entry.insert(END, "\n------------------------------Decryption Operation------------------------------\n")
        triangulation_entry.insert(END, f"\nTHE ENCRYPTED CIPHER TEXT: {encrypted_text}\n")
        triangulation_entry.insert(END, f"\nCipher Text to Binary Blocks: {binary_blocks_str}\n")

        # Initialize the summary table header
        summary_table = "Option       | Generated Block      | ASCII Value            | ASCII Character        | Matching Status\n"
        summary_table += "-" * 80 + "\n"

        final_decrypted_text = ""

        # Iterate over each block and apply triangulation options
        for block_index, block in enumerate(blocks):
            triangulation_entry.insert(END, f"\nBlock {block_index + 1}: {block}\n")

            for option in ['001', '010', '011', '100']:
                triangulation_entry.insert(END, f"\nOption {option}:\n")
                
                # Apply triangulation to block and get results
                result, generated_target_block, msb, lsb, encrypted_stream = apply_triangulation_to_block(block, option, block_index)
                triangulation_entry.insert(END, result)

                # Matching logic for XOR results comparison
                xor_result = xor_results[block_index]
                if generated_target_block == xor_result:
                    matching_status = "MATCHED"
                    
                    # XOR with the session key segment
                    session_key_segment = key_segments[block_index % len(key_segments)]
                    resultant_xor = ''.join(str(int(generated_target_block[i]) ^ int(session_key_segment[i])) for i in range(8))
                    
                    # Convert resultant XOR to ASCII
                    ascii_value = int(resultant_xor, 2)
                    ascii_character = chr(ascii_value)
                    final_decrypted_text += ascii_character

                    triangulation_entry.insert(END, f"RESULTANT XOR BETWEEN MATCH BLOCK & SESSION KEY: {resultant_xor}\n")
                else:
                    matching_status = "NOT MATCHED"

                # Add result to the summary table
                ascii_value = int(generated_target_block, 2)
                ascii_character = chr(ascii_value)
                summary_table += f"{option:<12} | {generated_target_block:<20} | {ascii_value:<25} | {ascii_character:<20} | {matching_status}\n"

        # After iterating over all blocks, display the final decrypted text
        triangulation_entry.insert(END, f"\nFINAL DECRYPTED TEXT: \"{final_decrypted_text}\"\n")
        triangulation_entry.insert(END, "Decryption Successfully Executed\n")
        triangulation_entry.insert(END, f"Session Key (128-bit in Decimal): {stored_key}\n")

        # Print the summary table
        triangulation_entry.insert(END, "\nSummary Table:\n")
        triangulation_entry.insert(END, summary_table)

        triangulation_entry.see(END)  # Ensure the latest content is visible

        # Calculate decryption time and update labels
        end_time = time.time()
        decryption_time = end_time - start_time
        total_decryption_time += decryption_time
        update_time_labels()
        

        # Destroy the key after successful decryption
        stored_key = None

    except Exception as e:
        messagebox.showerror('Error', str(e))

"""

def decrypt():
    global total_decryption_time, stored_key, final_cipher_text, used_options, original_source_blocks, xor_results
    
    try:
        # Example key input (Replace this with actual key input logic if necessary)
        key = int(key_entry.get())  # Assuming the key is entered in decimal format

        # Validate the key with the stored key
        if key != stored_key:
            stored_key = None  # Destroy the stored key on a wrong key attempt
            messagebox.showerror('Authentication Error', 'Wrong key provided. The key has been destroyed.')
            return

        start_time = time.time()

        # Convert the stored session key (decimal) to binary
        key_binary = f'{stored_key:0128b}'  # Convert stored key to a 128-bit binary string
        print(f"Session Key (128-bit Binary): {key_binary}")

        # Split the 128-bit session key into 8-bit segments
        key_segments = [key_binary[i:i+8] for i in range(0, 128, 8)]
        print(f"Session Key Segments (8-bit): {key_segments}")

        # Use the stored encrypted cipher text directly
        encrypted_text = final_cipher_text

        # Convert the encrypted text to binary stream
        encrypted_binary = text_to_binary(encrypted_text)

        # Divide the encrypted binary stream into 8-bit blocks
        blocks = divide_into_blocks(encrypted_binary)

        # Debugging output
        print(f"Encrypted text: {encrypted_text}")
        print(f"Encrypted binary: {encrypted_binary}")
        print(f"Blocks: {blocks}")

        # Create a string for binary blocks separated by '/'
        binary_blocks_str = '/'.join(blocks)

        # Display the heading and encrypted cipher text
        triangulation_entry.insert(END, "\n !!!!----------------------------------Decryption Operation----------------------------------!!!! \n")
        triangulation_entry.insert(END, f"\nTHE ENCRYPTED CIPHER TEXT: {encrypted_text}\n")
        triangulation_entry.insert(END, f"\nCipher Text to Binary Blocks: {binary_blocks_str}\n")

        # Initialize the summary table header
        summary_table = "Option       | Generated Block      | ASCII Value            | ASCII Character        | Matching Status\n"
        summary_table += "-" * 80 + "\n"

        final_decrypted_text = ""

        # Iterate over each block and apply triangulation options
        for block_index, block in enumerate(blocks):
            triangulation_entry.insert(END, f"\nBlock {block_index + 1}: {block}\n")

            match_found = False  # Flag to track if a match is already found for the current block

            for option in ['001', '010', '011', '100']:
                triangulation_entry.insert(END, f"\nOption {option}:\n")
                
                # Apply triangulation to block and get results
                result, generated_target_block, msb, lsb, encrypted_stream = apply_triangulation_to_block(block, option, block_index)
                triangulation_entry.insert(END, result)

                # Matching logic for XOR results comparison
                xor_result = xor_results[block_index]
                if generated_target_block == xor_result:
                    if not match_found:  # Process only the first match found
                        match_found = True  # Set the flag to True to avoid further matches
                        matching_status = "MATCHED"
                    
                        # XOR with the session key segment
                        session_key_segment = key_segments[block_index % len(key_segments)]
                        resultant_xor = ''.join(str(int(generated_target_block[i]) ^ int(session_key_segment[i])) for i in range(8))
                    
                        # Convert resultant XOR to ASCII
                        ascii_value = int(resultant_xor, 2)
                        ascii_character = chr(ascii_value)
                        final_decrypted_text += ascii_character

                        triangulation_entry.insert(END, f"RESULTANT XOR BETWEEN MATCH BLOCK & SESSION KEY: {resultant_xor}\n")
                    
                    else:
                        triangulation_entry.insert(END, "Skipping further matches for this block.\n")
                else:
                    matching_status = "NOT MATCHED"

                # Add result to the summary table
                ascii_value = int(generated_target_block, 2)
                ascii_character = chr(ascii_value)
                summary_table += f"{option:<12} | {generated_target_block:<20} | {ascii_value:<25} | {ascii_character:<20} | {matching_status}\n"









        # Create a string for source block characters separated by ' " " '
        #source_blocks_chars_str = ' '.join(final_decrypted_text)
        #matching_characters_str = ' '.join(final_decrypted_text)  # String of matching characters
        source_blocks_chars_str = ', '.join(f'"{char}"' for char in final_decrypted_text)




        # After iterating over all blocks, display the final decrypted text
        
        #triangulation_entry.insert(END, f"\nFINAL DECRYPTED TEXT: \"{final_decrypted_text}\"\n")
        #triangulation_entry.insert(END, f"\n !!----------FINAL DECRYPTED CHARACTER----------!! : \n\n{source_blocks_chars_str}\n\n")
        #triangulation_entry.insert(END, f"\n !!--------FINAL_DECRYPTED_TEXT--------!! :\n\n{final_decrypted_text}\n\n")
        triangulation_entry.insert(END, "Decryption Successfully Executed\n")
        triangulation_entry.insert(END, f"Session Key (128-bit in Decimal): {stored_key}\n")
        #triangulation_entry.insert(END, f"\n !!----------FINAL DECRYPTED CHARACTER----------!! : \n\n{source_blocks_chars_str}\n\n")
        #triangulation_entry.insert(END, f"\n !!--------FINAL_DECRYPTED_TEXT--------!! :\n\n{final_decrypted_text}\n\n")

        # Print the summary table
        triangulation_entry.insert(END, "\nSummary Table:\n")
        triangulation_entry.insert(END, summary_table)
        triangulation_entry.insert(END, f"\n !!----------FINAL DECRYPTED CHARACTER----------!! : \n\n{source_blocks_chars_str}\n\n")
        triangulation_entry.insert(END, f"\n !!--------FINAL_DECRYPTED_TEXT--------!! :\n\n{final_decrypted_text}\n\n")
        

        triangulation_entry.see(END)  # Ensure the latest content is visible

        # Calculate decryption time and update labels
        end_time = time.time()
        decryption_time = end_time - start_time
        total_decryption_time += decryption_time
        update_time_labels()
        

        # Destroy the key after successful decryption
        stored_key = None

    except Exception as e:
        messagebox.showerror('Error', str(e))



def is_valid_binary(string):
    return all(c in '01' for c in string)





def apply_triangulation_to_block(bit_stream, option, block_index):
    if not bit_stream or not all(c in '01' for c in bit_stream):
        raise ValueError("Invalid bit stream provided.")
    # Initialize the triangle with the first row being the original bit stream
    triangle = [list(bit_stream)]
    # Prepare the output list with the first iteration
    local_output = [f"Block-{block_index + 1} Iteration-1 {' '.join(list(bit_stream))}"]

    # Perform the triangulation iterations
    for iteration_index in range(len(bit_stream) - 1):
        new_iteration = []
        for j in range(len(triangle[iteration_index]) - 1):
            # Compute the new bit as the XOR of two adjacent bits
            new_bit = str(int(triangle[iteration_index][j]) ^ int(triangle[iteration_index][j + 1]))
            new_iteration.append(new_bit)
        # Append the new iteration to the triangle
        triangle.append(new_iteration)
        # Append the current iteration to the output
        local_output.append(f"Iteration-{iteration_index + 2} {' '.join(new_iteration)}")

    # Determine the encrypted stream based on the option
    if option == '001':
        encrypted_stream = ''.join(row[0] for row in triangle)
        arrow = "↓ (MSB source to last | 001)"
    elif option == '010':
        encrypted_stream = ''.join(row[0] for row in reversed(triangle))
        arrow = "↑ (MSB last to source | 010)"
    elif option == '011':
        encrypted_stream = ''.join(row[-1] for row in triangle)
        arrow = "↓ (LSB source to last | 011)"
    else:  # '100'
        encrypted_stream = ''.join(row[-1] for row in reversed(triangle))
        arrow = "↑ (LSB last to source | 100)"
    
    # Add the arrow description to the appropriate line
    local_output[0] += " " + arrow

    # Determine MSB and LSB
    msb = encrypted_stream[0] if encrypted_stream else "N/A"
    lsb = encrypted_stream[-1] if encrypted_stream else "N/A"
    
    # Append final results to the output
    local_output.append(f"\nGenerated Target Block: {encrypted_stream} \n")
    local_output.append(f"MSB: {msb} | LSB: {lsb}")

    # Return the full output and encrypted stream, along with the iteration details
    return '\n'.join(local_output[:-1]), encrypted_stream, msb, lsb, encrypted_stream





def reset():
    key_entry.delete(0, END)
    key_entry.focus()
    plaintext_text.delete("1.0", END)
    triangulation_entry.delete("1.0", END)

def iexit():
    iexit = messagebox.askyesno("Triangulation Encryption/Decryption", "Confirm if you want to exit")
    if iexit > 0:
        root.destroy()
        return
    
def record_voice():
    def record():
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Recording started. Speak now...")
                audio = r.listen(source, timeout=5)  # Timeout set to 5 seconds
            print("Recording ended.")
            text = r.recognize_google(audio)
            root.after(0, lambda: plaintext_text.insert(END, text))
        except sr.WaitTimeoutError:
            print("Recording timeout. No speech detected.")
            messagebox.showinfo('Info', 'Recording timeout. No speech detected.')
        except Exception as e:
            print("Error:", str(e))
            messagebox.showerror('Error', str(e))

    threading.Thread(target=record).start()

#def update_time_labels():
#    encryption_time_label.config(text=f"Encryption Time: {total_encryption_time:.2f} seconds")
#    decryption_time_label.config(text=f"Decryption Time: {total_decryption_time:.2f} seconds")



def update_time_labels():
    encryption_time_label.config(text=f"Encryption Time: {total_encryption_time:.6f} seconds")
    decryption_time_label.config(text=f"Decryption Time: {total_decryption_time:.6f} seconds")
    

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary_str):
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))






import random



def binary_to_ascii_numeric(binary_string):
    """ Convert a binary string to ASCII numeric value. """
    ascii_value = int(binary_string, 2)  # Convert binary to decimal integer
    return ascii_value
    

def ascii_to_cipher(ascii_value):
    """ Convert an ASCII value to cipher text. Replace this with actual cipher implementation. """
    # Example: Simple conversion for demonstration; replace with your cipher logic
    return chr(ascii_value)  # Convert ASCII value to character






original_source_blocks = []

def triangular_encrypt_block(bit_stream):
    global final_cipher_text, original_source_blocks # Ensure the global variable is used
    triangle_output = []
    all_encrypted_streams = []  # List to collect encrypted streams of each block
    all_original_streams = []   # List to collect original binary streams of each block
    all_cipher_texts = []       # List to collect cipher texts
    options_used = []           # List to collect the options used during encryption
    original_source_blocks = []

    # Define a function to apply triangulation to a bit stream
    def apply_triangulation(bit_stream, block_index):
        triangle = [list(bit_stream)]
        local_output = [f"Block-{block_index+1} Iteration-1 {' '.join(list(bit_stream))}"]
        for iteration_index in range(len(bit_stream) - 1):
            new_iteration = []
            for j in range(len(triangle[iteration_index]) - 1):
                new_bit = str(int(triangle[iteration_index][j]) ^ int(triangle[iteration_index][j + 1]))
                new_iteration.append(new_bit)
            triangle.append(new_iteration)
            local_output.append(f"Block-{block_index+1} Iteration-{iteration_index + 2} {' '.join(new_iteration)}")

        # Select a random option for each block
        option = random.choice(['001', '010', '011', '100'])
        options_used.append(option)  # Store the option used for this block
        
        arrow = ""
        if option == '001':
            encrypted_stream = ''.join(row[0] for row in triangle)
            arrow = "↓ (MSB source to last | 001)"
        elif option == '010':
            encrypted_stream = ''.join(row[0] for row in reversed(triangle))
            arrow = "↑ (MSB last to source | 010)"
        elif option == '011':
            encrypted_stream = ''.join(row[-1] for row in triangle)
            arrow = "↓ (LSB source to last | 011)"
        else:  # '100'
            encrypted_stream = ''.join(row[-1] for row in reversed(triangle))
            arrow = "↑ (LSB last to source | 100)"
        
        # Append the arrow to the appropriate output line
        for i in range(len(local_output)):
            if ((option == '001' or option == '011') and i == 0) or \
               ((option == '010' or option == '100') and i == len(local_output) - 1):
                local_output[i] += " " + arrow

        # Extract MSB and LSB from the encrypted_stream
        msb = encrypted_stream[0] if encrypted_stream else "N/A"
        lsb = encrypted_stream[-1] if encrypted_stream else "N/A"

        # Append the MSB and LSB information and generated target block
        local_output.append(f"\n Generated Target Block: {encrypted_stream} \n")
        local_output.append(f"MSB: {msb} | LSB: {lsb}")

        # Collect encrypted and original streams
        all_encrypted_streams.append(encrypted_stream)
        all_original_streams.append(bit_stream)

        return encrypted_stream, local_output

    # Divide the original bit stream into 8-bit blocks
    blocks = divide_into_blocks(bit_stream)


    # Store each original source block
    #original_source_blocks.append(bit_stream)  # Store the source block for later comparison


        # Store each original 8-bit block
    for block in blocks:
        original_source_blocks.append(block)  # Store each 8-bit block as a source block


    
    
    # Apply triangulation to each 8-bit block and collect outputs
    all_triangle_outputs = []
    for index, block in enumerate(blocks):
        encrypted_stream, local_output = apply_triangulation(block, index)
        all_triangle_outputs.extend(local_output)
    
    # Format and add the generated target blocks to the output
    formatted_target_blocks = "/".join(all_encrypted_streams)
    all_triangle_outputs.append(f" \n All Generated Target Blocks: {formatted_target_blocks} \n")

    # Create a table of source blocks, generated target blocks, ASCII Numeric Values, and Cipher Text
    table_output = []
    for idx, (source_block, encrypted_stream) in enumerate(zip(all_original_streams, all_encrypted_streams)):
        ascii_value = binary_to_ascii_numeric(encrypted_stream)
        cipher_text = ascii_to_cipher(ascii_value)
        table_output.append(
            f"Target Block {idx + 1} : {encrypted_stream} | Source Block: {source_block} | ASCII Numeric Value: {ascii_value} | Cipher Text: {cipher_text}"
        )
        all_cipher_texts.append(cipher_text)  # Collect cipher texts

    all_triangle_outputs.extend(table_output)
    
    # Add the final encrypted cipher text
    final_cipher_text = "".join(all_cipher_texts)
    all_triangle_outputs.append(f"\n THE ENCRYPTED CIPHER TEXT: {final_cipher_text} \n")


    # Return the final encrypted cipher text, all triangulation outputs, and options used
    return final_cipher_text, all_triangle_outputs, options_used









def reverse_triangulation(encrypted_stream, option):
    # Rebuild the triangle from the encrypted stream based on the selected option
    triangle = []
    
    if option == '001':  # MSB source to last
        triangle.append([bit for bit in encrypted_stream])
    elif option == '010':  # MSB last to source
        triangle.append([bit for bit in reversed(encrypted_stream)])
    elif option == '011':  # LSB source to last
        triangle.append([bit for bit in encrypted_stream])
    elif option == '100':  # LSB last to source
        triangle.append([bit for bit in reversed(encrypted_stream)])
    
    # Reverse the triangular operation
    for i in range(1, len(triangle[0])):
        new_row = []
        for j in range(len(triangle[i-1]) - 1):
            new_bit = str(int(triangle[i-1][j]) ^ int(triangle[i-1][j + 1]))
            new_row.append(new_bit)
        triangle.append(new_row)
    
    # The original bit stream is the last row of the triangle
    return ''.join(triangle[-1])


def get_option_for_block(block):
    # This function retrieves the option used during encryption based on the block's binary content
    # Implement this logic based on how the option was assigned in encryption
    if block[0] == '0':
        return '001'
    elif block[-1] == '0':
        return '010'
    elif block[0] == '1':
        return '011'
    else:
        return '100'















end_time = time.time()













def plot_graph():
    # Create a new figure
    fig, ax = plt.subplots()

    # Data for plotting
    labels = ['Encryption Time', 'Decryption Time']
    times = [total_encryption_time, total_decryption_time]

    # Plotting the data
    ax.bar(labels, times, color=['blue', 'green'])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Encryption and Decryption Times')

    # Embed the plot in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)



def show_graph():
    global chi_square_value, degrees_of_freedom  # Use global variables if they are defined globally

    # Create a new popup window
    graph_window = Toplevel(root)
    graph_window.title("Encryption and Decryption Time Graph")

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plotting times
    bars = ax1.bar(['Encryption', 'Decryption'], [total_encryption_time, total_decryption_time], color=['blue', 'red'])
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Encryption and Decryption Time Comparison')

    # Add detailed text to the right side of the graph
    detailed_text = (f"Encryption Time: {total_encryption_time:.6f} seconds\n"
                     f"Decryption Time: {total_decryption_time:.6f} seconds")

    # Hide x-axis and y-axis for the second subplot
    ax2.axis('off')
    ax2.text(0.5, 1.0, detailed_text, ha='center', va='top', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7))

    # Add chi-square value and degrees of freedom below the times
    text = (f"Chi-Square Value: {chi_square_value:.6f}\n"
            f"Degrees of Freedom: {degrees_of_freedom}")

    ax2.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7))

    # Adjust layout to ensure no overlap
    plt.tight_layout()

    # Create a canvas and add it to the popup window
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    # Optionally, add a button to close the graph window
    close_button = Button(graph_window, text="Close", command=graph_window.destroy)
    close_button.pack(pady=10)








def upload_file():
    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[
            ("All Supported Files", "*.txt *.com *.sys *.pdf *.exe *.dos *.cpp *.docx *.dll"),
            ("Text Files", "*.txt"),
            ("Executable Files", "*.exe"),
            ("PDF Files", "*.pdf"),
            ("C++ Source Files", "*.cpp"),
            ("Word Documents", "*.docx"),
            ("System Files", "*.sys *.com *.dll"),
            ("All Files", "*.*")  # Option to show all files
        ]
    )
    if file_path:  # Check if a file was selected
        with open(file_path, 'r', errors='ignore') as file:  # Ignore errors for unsupported formats
            file_content = file.read()  # Read the file content
            plaintext_text.delete(1.0, tk.END)  # Clear the "Enter Plain Text" section
            plaintext_text.insert(tk.END, file_content)  # Insert the file content into the text section


















#======================================================================================================

button_frame = Frame(root, bg='green')
button_frame.pack()






# Set the root window background to green
root.configure(bg='green')












Graph_button = Button(button_frame, font=('arial', 24, 'bold'), bg='blue', fg='white', width=5, text="Graph", command=show_graph)
Graph_button.pack(side=LEFT, padx=10)



Upload_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=6, text="Upload", command=upload_file )
Upload_button.pack(side=LEFT, padx=10)

Image_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=5, text="Image", command=encrypt )
Image_button.pack(side=LEFT, padx=10)

Video_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=5, text="Video", command=encrypt )
Video_button.pack(side=LEFT, padx=10)

Encryption_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=7, text="Encrypt", command=encrypt )
Encryption_button.pack(side=LEFT, padx=10)

Decryption_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=7, text="Decrypt", command=decrypt )
Decryption_button.pack(side=LEFT, padx=10)

Reset_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=5, text="Reset", command=reset )
Reset_button.pack(side=LEFT, padx=10)

Exit_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=3, text="Exit", command=iexit )
Exit_button.pack(side=LEFT, padx=10)

Record_button = Button(button_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', width=10, text="Record Voice", command=record_voice )
Record_button.pack(side=LEFT, padx=10)

#======================================================================================================

time_frame = Frame(root, bg='yellow')
time_frame.pack(pady=20)

encryption_time_label = Label(time_frame, font=('arial', 24, 'bold'), bg = 'yellow', fg = 'black', text="Encryption Time: 0.00 seconds")
encryption_time_label.pack(side=LEFT, padx=10)

decryption_time_label = Label(time_frame, font=('arial', 24, 'bold'), bg = 'yellow', fg = 'black', text="Decryption Time: 0.00 seconds")
decryption_time_label.pack(side=LEFT, padx=10)


#======================================================================================================

key_frame = Frame(root, bg='red')
key_frame.pack(pady=20)

key_label = Label(key_frame, font=('arial',24,'bold'), bg = 'red', fg = 'white', text="Enter Session Key:")
key_label.pack(side=LEFT, padx=10)
key_entry = Entry(key_frame, font=('arial',24,'bold'), width=12, justify='center', show="*")
key_entry.pack(side=LEFT, padx=10)

# Frame for plain text section
plain_frame = Frame(root, bg = 'blue')
plain_frame.pack(side=LEFT, pady=2)

plaintext_label = Label(plain_frame, font=('arial',20,'bold'), bg = 'blue', fg = 'white', text="Enter Plain Text:")
plaintext_label.pack(side=TOP, padx=1)

# Create a scrollbar for the plaintext_text
plaintext_scrollbar = Scrollbar(plain_frame)
plaintext_scrollbar.pack(side=RIGHT, fill=Y)

# Create the plaintext_text Text widget and link it with the scrollbar
plaintext_text = Text(plain_frame, font=('arial',20,'bold'), width=20, height=15, yscrollcommand=plaintext_scrollbar.set)
plaintext_text.pack(pady=2, padx=1)

# Configure the scrollbar to control the plaintext_text
plaintext_scrollbar.config(command=plaintext_text.yview)

# Frame for triangulation structure section
triangulation_frame = Frame(root, bg = 'blue')
triangulation_frame.pack(side=RIGHT, pady=2)

triangulation_label = Label(triangulation_frame, font=('arial',24,'bold'), bg = 'blue', fg = 'white', text="Triangulation Structure:")
triangulation_label.pack(side=TOP, padx=1)

# Create a scrollbar for the triangulation_entry
triangulation_scrollbar = Scrollbar(triangulation_frame)
triangulation_scrollbar.pack(side=RIGHT, fill=Y)

# Create the triangulation_entry Text widget and link it with the scrollbar
triangulation_entry = Text(triangulation_frame, font=('arial',12,'bold'), width=110, height=30, yscrollcommand=triangulation_scrollbar.set)
triangulation_entry.pack(pady=2, padx=1)

# Configure the scrollbar to control the triangulation_entry
triangulation_scrollbar.config(command=triangulation_entry.yview)

root.mainloop()
