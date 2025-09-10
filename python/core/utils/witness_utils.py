from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO


def read_usize(f: BinaryIO, usize_len: int | None = 8) -> int:
    """
    Read an unsigned integer of size `usize_len` bytes from a binary file object.

    Args:
        f (BinaryIO): Opened file in binary mode.
        usize_len (int, optional): Number of bytes to read
        (default is 8, for 64-bit systems).

    Returns:
        int: The unpacked unsigned integer.
    """
    return struct.unpack("<Q", f.read(usize_len))[0]


def read_u256(f: BinaryIO) -> int:
    """
    Read a 256-bit unsigned integer (U256) from a binary file in little-endian format.

    Args:
        f (BinaryIO): Opened file in binary mode.

    Returns:
        int: The 256-bit integer.
    """
    return int.from_bytes(f.read(32), "little")


def read_field_elements(f: BinaryIO, count: int) -> list[int]:
    """
    Read a sequence of 32-byte field elements from a binary file.

    Args:
        f (BinaryIO): Opened file in binary mode.
        count (int): Number of 32-byte elements to read.

    Returns:
        list[int]: List of integers representing the field elements.
    """
    return [int.from_bytes(f.read(32), "little") for _ in range(count)]


def load_witness(path: str) -> dict:
    """
    Load witness data from a binary file and return it in structured form.

    Args:
        path (str): Path to the binary witness file.

    Returns:
        dict: Dictionary containing:
            - num_witnesses (int)
            - num_inputs_per_witness (int)
            - num_public_inputs_per_witness (int)
            - modulus (int)
            - witnesses (list of dicts with 'inputs' and 'public_inputs')
    """
    with Path(path).open("rb") as f:
        num_witnesses = read_usize(f)
        num_inputs = read_usize(f)
        num_public_inputs = read_usize(f)
        modulus = read_u256(f)

        total = num_witnesses * (num_inputs + num_public_inputs)
        values = read_field_elements(f, total)

    # Reshape into witnesses
    witnesses = []
    offset = 0
    for _ in range(num_witnesses):
        inputs = values[offset : offset + num_inputs]
        public_inputs = values[
            offset + num_inputs : offset + num_inputs + num_public_inputs
        ]
        witnesses.append({"inputs": inputs, "public_inputs": public_inputs})
        offset += num_inputs + num_public_inputs

    return {
        "num_witnesses": num_witnesses,
        "num_inputs_per_witness": num_inputs,
        "num_public_inputs_per_witness": num_public_inputs,
        "modulus": modulus,
        "witnesses": witnesses,
    }


def to_field_repr(value: int, modulus: int) -> int:
    """
    Convert a signed integer to its field representation modulo `modulus`.

    Args:
        value (int): Integer to convert.
        modulus (int): Field modulus.

    Returns:
        int: Least field representation of the integer, ensuring a non-negative result.
    """
    if value < 0:
        return modulus + value
    return value % modulus


def compare_witness_to_io(
    witnesses: dict,
    expected_inputs: dict,
    expected_outputs: dict,
    modulus: int,
) -> bool:
    """
    Compare the public inputs of the first witness against expected inputs and outputs.

    Accounts for negative numbers by representing them in the field as
    `modulus - abs(value)`.

    Args:
        witnesses (dict): Witness data as returned by `load_witness`.
        expected_inputs (dict):
            Dictionary containing key "input"
            mapping to a list of expected input integers.
        expected_outputs (dict):
            Dictionary containing key "output"
            mapping to a list of expected output integers.
        modulus (int): Field modulus.

    Returns:
        bool:
        True if the witness public inputs match the expected inputs and outputs,
        False otherwise.
    """

    # Convert expectations into field form
    inputs_list = expected_inputs.get("input", [])
    outputs_list = expected_outputs.get("output", [])

    expected_inputs_mod = [to_field_repr(v, modulus) for v in inputs_list]
    expected_outputs_mod = [to_field_repr(v, modulus) for v in outputs_list]

    n_inputs = len(expected_inputs_mod) + len(expected_outputs_mod)

    witness = witnesses["witnesses"][0]["public_inputs"]

    if n_inputs != len(witness):
        return False

    actual_inputs = witness[: len(expected_inputs_mod)]
    actual_outputs = witness[len(expected_inputs_mod) :]

    # Compare
    if actual_inputs != expected_inputs_mod or actual_outputs != expected_outputs_mod:
        return False

    return True


if __name__ == "__main__":
    import time

    start_time = time.time()
    w = load_witness("./artifacts/lenet/witness.bin")
    end_time = time.time()
    print("Modulus:", w["modulus"])  # noqa: T201
    print("First witness inputs:", w["witnesses"][0]["inputs"][0])  # noqa: T201
    print(  # noqa: T201
        "First witness public inputs:",
        w["witnesses"][0]["public_inputs"][0],
    )

    print(len(w["witnesses"][0]["public_inputs"]))  # noqa: T201
    print((w["witnesses"][0]["public_inputs"][0] - w["modulus"]) / 2**18)  # noqa: T201
    elapsed = end_time - start_time

    print("time taken: ", elapsed)  # noqa: T201
