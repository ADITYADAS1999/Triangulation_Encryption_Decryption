
import hashlib

def hash_file(filename):
    hasher = hashlib.sha256()
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def compare_files(file1, file2):
    if hash_file(file1) == hash_file(file2):
        print("The files are identical.")
    else:
        print("The files are different.")

# Usage
compare_files('test_file_02.txt', 'final.txt')
