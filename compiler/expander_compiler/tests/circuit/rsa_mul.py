def multiply_mod_254bit_constraint(a_chunks, b_chunks, r_chunks, chunk_size=120):
    """
    Compute (a * b) mod r where:
    - a, b, r are stored as arrays of 17 120-bit chunks
    - All intermediate calculations must stay under 254 bits
    - Result should be in the same format (17 120-bit chunks)

    Parameters:
    a_chunks: List[int] - 17 elements, each ≤ 2^120
    b_chunks: List[int] - 17 elements, each ≤ 2^120
    r_chunks: List[int] - 17 elements, each ≤ 2^120
    """
    n = len(a_chunks)  # Should be 17
    assert len(a_chunks) == len(b_chunks) == len(r_chunks) == 17

    # Maximum value in each chunk
    MAX_CHUNK = (1 << chunk_size)

    def single_chunk_mult(a_i, b_j):
        """
        Multiply two 120-bit chunks, ensuring result stays under 254 bits
        Returns (high, low) tuple where high is the overflow into next chunk
        """
        product = a_i * b_j
        low = product & (MAX_CHUNK - 1)
        high = product >> chunk_size
        return high, low

    # Initialize result array (need extra space for overflow)
    result = [0] * (2 * n)

    # Perform multiplication with careful overflow handling
    for i in range(n):
        for j in range(n):
            # Multiply individual chunks
            high, low = single_chunk_mult(a_chunks[i], b_chunks[j])

            # Position in result
            pos = i + j

            # Add low part
            result[pos] += low

            # Handle overflow from low part
            if result[pos] >= MAX_CHUNK:
                result[pos + 1] += result[pos] >> chunk_size
                result[pos] &= (MAX_CHUNK - 1)

            # Add high part to next chunk
            result[pos + 1] += high

            # Handle overflow from high part
            if result[pos + 1] >= MAX_CHUNK:
                result[pos + 2] += result[pos + 1] >> chunk_size
                result[pos + 1] &= (MAX_CHUNK - 1)

    # Now we need to reduce modulo r
    def reduce_mod_r():
        """
        Reduce the result modulo r, maintaining chunk structure
        This is a simplified Barrett reduction adapted for our chunk structure
        """
        # Convert chunks to a more manageable form for reduction
        temp_result = result.copy()

        while any(temp_result[n:]):  # While there are any nonzero high chunks
            # Find the highest nonzero chunk
            highest_chunk = 2 * n - 1
            while highest_chunk >= n and temp_result[highest_chunk] == 0:
                highest_chunk -= 1

            if highest_chunk < n:
                break

            # Perform the reduction
            shift = highest_chunk - n + 1
            for i in range(n):
                if r_chunks[i] != 0:
                    for j in range(n):
                        if i + j + shift < 2 * n:
                            temp_result[i + j + shift] -= (
                                (temp_result[highest_chunk] * r_chunks[i])
                                >> (chunk_size * (n - j - 1))
                            ) & (MAX_CHUNK - 1)

            # Normalize chunks
            for i in range(2 * n - 1):
                while temp_result[i] < 0:
                    temp_result[i] += MAX_CHUNK
                    temp_result[i + 1] -= 1
                while temp_result[i] >= MAX_CHUNK:
                    temp_result[i] -= MAX_CHUNK
                    temp_result[i + 1] += 1

        return temp_result[:n]

    return reduce_mod_r()

def test_complex_case():
    # Create a more complex test case with larger numbers

    # Helper function to create chunks from a large number
    def number_to_chunks(num, chunk_size=120, num_chunks=17):
        chunks = []
        mask = (1 << chunk_size) - 1
        for _ in range(num_chunks):
            chunks.append(num & mask)
            num >>= chunk_size
        return chunks

    # Test with some large prime-like numbers
    # Using numbers that will exercise multiple chunks

    # Create a large number for testing (about 1000 bits set)
    a = (1 << 1000) - (1 << 500) + (1 << 200) - (1 << 100) + 12345
    b = (1 << 900) - (1 << 400) + (1 << 300) - (1 << 50) + 67890
    r = (1 << 1024) - (1 << 512) + (1 << 256) - (1 << 128) + 11111

    # Convert to chunks
    a_chunks = number_to_chunks(a)
    b_chunks = number_to_chunks(b)
    r_chunks = number_to_chunks(r)

    # Perform the modular multiplication
    result_chunks = multiply_mod_254bit_constraint(a_chunks, b_chunks, r_chunks)

    # Convert result back to number for verification
    def chunks_to_number(chunks, chunk_size=120):
        result = 0
        for i, chunk in enumerate(chunks):
            result += chunk << (i * chunk_size)
        return result

    # Calculate expected result using regular Python arithmetic
    expected = (a * b) % r
    actual = chunks_to_number(result_chunks)

    print("Test with large numbers:")
    print(f"a (bits): {a.bit_length()}")
    print(f"b (bits): {b.bit_length()}")
    print(f"r (bits): {r.bit_length()}")
    print("a:", a)
    print("b:", b)
    print("r:", r)
    print("expected:", expected)
    print("actual:", actual)
    print("\nFirst few chunks of a:", a_chunks[:3])
    print("First few chunks of b:", b_chunks[:3])
    print("First few chunks of r:", r_chunks[:3])
    print("\nFirst few chunks of result:", result_chunks[:3])
    print(f"\nExpected result (bits): {expected.bit_length()}")
    print(f"Actual result (bits): {actual.bit_length()}")
    print(f"Results match: {expected == actual}")

    # Additional verification
    print("\nVerification that no chunk exceeds 120 bits:")
    max_chunk_size = max(chunk.bit_length() for chunk in result_chunks)
    print(f"Maximum chunk size in result: {max_chunk_size} bits")
    print(f"All chunks within 120-bit limit: {max_chunk_size <= 120}")

# Run the test
test_complex_case()
