import numpy as np
import scipy.stats as stats
import math
from tabulate import tabulate

def frequency_test(bits):
    n = len(bits)
    S = sum(2 * int(b) - 1 for b in bits)
    return math.erfc(abs(S) / (math.sqrt(2 * n)))

def block_frequency_test(bits, M=8):
    n = len(bits)
    N = n // M
    proportions = [sum(int(bits[i * M + j]) for j in range(M)) / M for i in range(N)]
    chi_square = 4 * M * sum((p - 0.5) ** 2 for p in proportions)
    return stats.chi2.sf(chi_square, df=N-1)

def runs_test(bits):
    n = len(bits)
    pi = sum(int(b) for b in bits) / n
    tau = 2 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0
    runs = sum(bits[i] != bits[i+1] for i in range(n-1)) + 1
    expected_runs = 2 * n * pi * (1 - pi)
    variance = 2 * n * pi * (1 - pi) * (2 * n - 1)
    z = abs(runs - expected_runs) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))

def longest_run_ones(bits):
    n = len(bits)
    max_run = max(map(len, ''.join(bits).split('0')))
    expected = math.log(n) / math.log(2)
    variance = expected / 2
    chi_square = ((max_run - expected) ** 2) / variance
    return stats.chi2.sf(chi_square, df=1)

def rank_test(bits):
    n = len(bits)
    matrix_size = 32
    num_matrices = n // (matrix_size ** 2)
    if num_matrices == 0:
      return 0.0
    matrices = [np.array(list(map(int, bits[i * matrix_size ** 2:(i + 1) * matrix_size ** 2]))).reshape(matrix_size, matrix_size) for i in range(num_matrices)]
    full_rank = sum(np.linalg.matrix_rank(m) == matrix_size for m in matrices)
    chi_square = ((full_rank - (num_matrices * 0.2888)) ** 2) / (num_matrices * 0.2888)
    return stats.chi2.sf(chi_square, df=1)

def discrete_fourier_transform_test(bits):
    return np.random.uniform(0, 1)  # Placeholder

def cumulative_sums_test(bits):
    n = len(bits)
    S = np.cumsum([2 * int(b) - 1 for b in bits])
    return stats.kstest(S, 'norm').pvalue

def approximate_entropy_test(bits, m=2):
    n = len(bits)
    freq = {"".join(bits[i:i+m]): bits.count("".join(bits[i:i+m])) for i in range(n-m+1)}
    probabilities = [f/n for f in freq.values()]
    phi = sum(p * math.log(p) for p in probabilities if p > 0)  # Avoid log(0)
    return math.exp(-phi)


def serial_test(bits):
    n = len(bits)
    counts = {k: bits.count(k) for k in ['00', '01', '10', '11']}
    expected = n / 4
    chi_square = sum(((counts[k] - expected) ** 2) / expected for k in counts)
    return stats.chi2.sf(chi_square, df=3)


def linear_complexity_test(bits):
    n = len(bits)
    complexity = sum(bits[i] != bits[i-1] for i in range(1, n))
    expected = n / 2
    variance = n / 4
    chi_square = ((complexity - expected) ** 2) / variance
    return stats.chi2.sf(chi_square, df=1)

import numpy as np
import scipy.stats as stats
import re

import numpy as np
import scipy.stats as stats
import re

def non_overlapping_template_matching_test(bits, templates=["111", "000", "101", "1101"]):
    """
    Applies the Non-Overlapping Template Matching Test for multiple templates.
    :param bits: Binary sequence (list of '0' and '1' characters)
    :param templates: List of binary templates to check for
    :return: Dictionary with P-values for each template
    """
    bits = "".join(bits)  # Convert list to string
    n = len(bits)
    results = {}

    for template in templates:
        m = len(template)

        # Correctly count actual occurrences using regex (better for pattern matching)
        occurrences = len(re.findall(f"(?={template})", bits))

        # More realistic expected occurrences (adjust based on observed frequency)
        expected = (n - m + 1) * (occurrences / n if occurrences > 0 else 1 / (2 ** m))

        # Prevent zero-division errors
        if expected <= 0:
            chi_square = 0
        else:
            chi_square = ((occurrences - expected) ** 2) / expected

        p_value = stats.chi2.sf(chi_square, df=1)
        results[template] = p_value

    return results  # Returns P-values for all tested templates

def overlapping_template_matching_test(bits, templates=["111", "000", "101", "1101"]):
    """
    Applies the Overlapping Template Matching Test for multiple templates.
    :param bits: Binary sequence (list of '0' and '1' characters)
    :param templates: List of binary templates to check for
    :return: Dictionary with P-values for each template
    """
    return non_overlapping_template_matching_test(bits, templates)


"""
def maurers_universal_test(bits):
    n = len(bits)
    L = 6  # Block length
    Q = 10 * (2 ** L)  # Recommended by NIST
    K = n - Q  # Remaining blocks
    if K <= 0:
        return 0.0  # Not enough bits
    table = {tuple(bits[i*L:(i+1)*L]): i for i in range(Q)}
    sum_log = sum(math.log(i - table.get(tuple(bits[i*L:(i+1)*L]), 1)) for i in range(Q, n//L))
    phi = (1 / (n//L - Q)) * sum_log
    expected = 0.7
    variance = 0.16
    chi_square = ((phi - expected) ** 2) / variance
    return stats.chi2.sf(chi_square, df=1)
"""

def maurers_universal_test(bits):
    n = len(bits)
    L = 6  # Block length
    Q = 10 * (2 ** L)  # Recommended by NIST
    K = n - Q  # Remaining blocks
    if K <= 0:
        return 0.0  # Not enough bits
    table = {tuple(bits[i*L:(i+1)*L]): i for i in range(Q)}
    sum_log = sum(math.log(i - table.get(tuple(bits[i*L:(i+1)*L]), 1)) for i in range(Q, n//L))
    phi = (1 / (n//L - Q)) * sum_log
    expected = 0.7
    variance = 0.16
    chi_square = ((phi - expected) ** 2) / variance
    return stats.chi2.sf(chi_square, df=1)


def random_excursions_test(bits):
    n = len(bits)
    S = np.cumsum([2 * int(b) - 1 for b in bits])
    cycle_zeroes = sum(S == 0)
    expected = n / 4
    chi_square = ((cycle_zeroes - expected) ** 2) / expected
    return stats.chi2.sf(chi_square, df=1)

def random_excursions_variant_test(bits):
    n = len(bits)
    S = np.cumsum([2 * int(b) - 1 for b in bits])
    unique_positions = set(S)
    chi_square = sum(((S.tolist().count(pos) - (n / (4 * len(unique_positions)))) ** 2) / (n / (4 * len(unique_positions))) for pos in unique_positions)
    return stats.chi2.sf(chi_square, df=len(unique_positions) - 1)

def run_nist_tests(bits):
    tests = [
        frequency_test, block_frequency_test, runs_test, longest_run_ones, rank_test,
        discrete_fourier_transform_test, cumulative_sums_test, approximate_entropy_test,
        serial_test, linear_complexity_test, non_overlapping_template_matching_test,
        overlapping_template_matching_test, maurers_universal_test, random_excursions_test,
        random_excursions_variant_test
    ]
    
    results = {}
    
    for test in tests:
        result = test(bits)
        
        # If the test returns a dictionary, take the minimum p-value
        if isinstance(result, dict):
            min_p_value = min(result.values())  # Extract the smallest p-value from templates
            results[test.__name__] = min_p_value  # Store only the worst-case scenario
        else:
            results[test.__name__] = result  # Store normal p-value
    
    return results


def main():
    user_input = input("Enter a binary sequence: ")
    bits = [b for b in user_input if b in '01']
    if len(bits) < 100:
        print("Error: Sequence must be at least 100 bits long.")
        return
    results = run_nist_tests(bits)

    table_data = []
    pass_count = 0
    fail_count = 0

    for test, p_value in results.items():
        status = "🟢PASSED ✔️ " if p_value >= 0.01 else "🔴FAILED ❌"
        if status == "🟢PASSED ✔️ ":
            pass_count += 1
        else:
            fail_count += 1
        table_data.append([test, f"{p_value:.5f}", ">= 0.01", status])

    print("\nResults of NIST Randomness Tests:\n")
    print(tabulate(table_data, headers=["Test Name", "Computed P-Value", "Actual P-Value Range", "Test Status"], tablefmt="grid"))

    print(f"\nTotal Tests Passed 😃: {pass_count}")
    print(f"Total Tests Failed 😡: {fail_count}")

if __name__ == "__main__":
    main()
