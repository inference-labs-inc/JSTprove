from __future__ import annotations

import io
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import zstandard

if TYPE_CHECKING:
    from collections.abc import Callable

from python.core.utils.errors import ProofSystemNotImplementedError
from python.core.utils.helper_functions import ZSTD_MAGIC, ZKProofSystems

MIN_PUBLIC_INPUTS_LENGTH = 2  # scale_base + scale_exponent
MAX_SCALE_BASE = 256
MAX_SCALE_EXPONENT = 64


def _flatten(x: list | tuple | float) -> list:
    if not isinstance(x, (list, tuple)):
        return [x]
    result = []
    for item in x:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _open_maybe_compressed(path: str) -> BinaryIO:
    with Path(path).open("rb") as raw:
        magic = raw.read(4)
        raw.seek(0)
        if magic == ZSTD_MAGIC:
            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(raw)
            return io.BytesIO(reader.read())
        return io.BytesIO(raw.read())


# -------------------------
# Base Witness Loader
# -------------------------
class WitnessLoader(ABC):
    def __init__(self: WitnessLoader, path: str) -> None:
        self.path = path

    @abstractmethod
    def load_witness(self: WitnessLoader) -> dict:
        """Load witness data from file."""

    @abstractmethod
    def compare_witness_to_io(
        self: WitnessLoader,
        witnesses: dict,
        expected_inputs: dict,
        expected_outputs: dict,
        modulus: int,
    ) -> bool:
        """Compare witness to expected I/O."""


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
    return [read_u256(f) for _ in range(count)]


def to_field_repr(value: int, modulus: int) -> int:
    """
    Convert a signed integer to its field representation modulo `modulus`.

    Args:
        value (int): Integer to convert.
        modulus (int): Field modulus.

    Returns:
        int: Least field representation of the integer, ensuring a non-negative result.
    """
    return value % modulus


def from_field_repr(value: int, modulus: int) -> int:
    """
    Convert a field element back to a signed integer.

    Values greater than modulus/2 are treated as negative.

    Args:
        value (int): Field element.
        modulus (int): Field modulus.

    Returns:
        int: Signed integer representation.
    """
    if value > modulus // 2:
        return value - modulus
    return value


def scale_to_field(
    values: list,
    scale_base: int,
    scale_exp: int,
    modulus: int,
) -> list[int]:
    """
    Scale values and convert to field representation.

    Args:
        values: List of numeric values to scale.
        scale_base: Base for scaling (e.g., 2).
        scale_exp: Exponent for scaling (e.g., 18 for 2^18).
        modulus: Field modulus.

    Returns:
        List of field elements.
    """
    if scale_base <= 0 or scale_exp <= 0:
        return [int(v) % modulus for v in values]
    scale = scale_base**scale_exp
    return [round(v * scale) % modulus for v in values]


def descale_outputs(
    outputs: list[int],
    scale_base: int,
    scale_exp: int,
) -> list[float]:
    """
    Descale output values back to original range.

    Args:
        outputs: List of signed integer outputs.
        scale_base: Base for scaling.
        scale_exp: Exponent for scaling.

    Returns:
        List of descaled float values.
    """
    if scale_base <= 0 or scale_exp <= 0:
        return [float(v) for v in outputs]
    scale = scale_base**scale_exp
    return [v / scale for v in outputs]


def compare_field_values(
    expected: list[int],
    actual: list[int],
    modulus: int,
    tolerance: int = 1,
) -> bool:
    """
    Compare field values with tolerance for rounding errors.

    Args:
        expected: Expected field values.
        actual: Actual field values.
        modulus: Field modulus.
        tolerance: Maximum allowed difference (default 1).

    Returns:
        True if all values match within tolerance.
    """
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual, strict=True):
        diff = (e - a) % modulus
        if diff > tolerance and diff < modulus - tolerance:
            return False
    return True


def extract_io_from_witness(
    witness_data: dict,
    num_inputs: int,
    num_outputs: int,
) -> dict | None:
    """
    Extract inputs, outputs, and scaling parameters from witness public inputs.

    The witness public_inputs layout is:
    [inputs..., outputs..., scale_base, scale_exp, weights... (if WAI)]

    Args:
        witness_data: Witness data as returned by load_witness().
        num_inputs: Number of input values in the public inputs.
        num_outputs: Number of output values in the public inputs.

    Returns:
        Dictionary containing:
            - inputs: Raw input field elements
            - raw_outputs: Raw output field elements (unsigned)
            - outputs: Output values converted to signed integers
            - rescaled_outputs: Outputs converted back to original scale
            - scale_base: Scaling base from witness
            - scale_exponent: Scaling exponent from witness
            - modulus: Field modulus
        Returns None if witness structure is invalid.
    """
    witnesses = witness_data.get("witnesses")
    if not isinstance(witnesses, list) or len(witnesses) == 0:
        return None

    first_witness = witnesses[0]
    if not isinstance(first_witness, dict):
        return None

    public_inputs = first_witness.get("public_inputs")
    if (
        not isinstance(public_inputs, list)
        or len(public_inputs) < MIN_PUBLIC_INPUTS_LENGTH
    ):
        return None

    modulus = witness_data.get("modulus")
    if modulus is None:
        return None

    if (
        not isinstance(num_inputs, int)
        or not isinstance(num_outputs, int)
        or num_inputs < 0
        or num_outputs < 0
        or num_inputs + num_outputs + 2 > len(public_inputs)
    ):
        return None

    inputs = public_inputs[:num_inputs]
    raw_outputs = public_inputs[num_inputs : num_inputs + num_outputs]
    scale_base = public_inputs[num_inputs + num_outputs]
    scale_exponent = public_inputs[num_inputs + num_outputs + 1]

    if scale_base > MAX_SCALE_BASE or scale_exponent > MAX_SCALE_EXPONENT:
        return None

    signed_outputs = [from_field_repr(v, modulus) for v in raw_outputs]
    rescaled_outputs = descale_outputs(signed_outputs, scale_base, scale_exponent)

    return {
        "inputs": inputs,
        "raw_outputs": raw_outputs,
        "outputs": signed_outputs,
        "rescaled_outputs": rescaled_outputs,
        "scale_base": scale_base,
        "scale_exponent": scale_exponent,
        "modulus": modulus,
    }


class ExpanderWitnessLoader(WitnessLoader):
    def load_witness(self: ExpanderWitnessLoader) -> dict:
        """
        Load witness data from a binary file and return it in structured form.

        Returns:
            dict: Dictionary containing:
                - num_witnesses (int)
                - num_inputs_per_witness (int)
                - num_public_inputs_per_witness (int)
                - modulus (int)
                - witnesses (list of dicts with 'inputs' and 'public_inputs')
        """
        path = self.path
        with _open_maybe_compressed(path) as f:
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

    def compare_witness_to_io(
        self: ExpanderWitnessLoader,
        witnesses: dict,
        expected_inputs: dict,
        expected_outputs: dict,
        modulus: int,
        scaling_function: Callable[[list[int], int, int], list[int]] | None = None,
    ) -> bool:
        """
        Compare the public inputs of the first witness
        against expected inputs and outputs.

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
            scaling_function
                (Callable[[list[int], int, int], list[int]] | None, optional):
                    Optional scaling function to apply to inputs.
                    Takes (inputs, scale_base, scale_exponent)
                    and returns scaled inputs. Defaults to None.

        Returns:
            bool:
            True if the witness public inputs match the expected inputs and outputs,
            False otherwise.
        """

        inputs_raw = expected_inputs.get("input", [])
        outputs_raw = expected_outputs.get("output", [])

        flat_inputs = _flatten(inputs_raw)
        flat_outputs = _flatten(outputs_raw)
        n_in = len(flat_inputs)
        n_out = len(flat_outputs)

        witness_list = witnesses.get("witnesses")
        if not witness_list:
            return False

        first = witness_list[0]
        if not isinstance(first, dict):
            return False
        witness = first.get("public_inputs")
        if not isinstance(witness, list):
            return False

        if len(witness) < n_in + n_out + 2:
            return False

        scale_base = witness[n_in + n_out]
        scale_exponent = witness[n_in + n_out + 1]

        if scale_base > MAX_SCALE_BASE or scale_exponent > MAX_SCALE_EXPONENT:
            return False

        if callable(scaling_function):
            scaled_inputs = scaling_function(flat_inputs, scale_base, scale_exponent)
        elif scale_base <= 0 or scale_exponent <= 0:
            scaled_inputs = [int(v) for v in flat_inputs]
        else:
            scale = scale_base**scale_exponent
            scaled_inputs = [round(v * scale) for v in flat_inputs]

        expected_inputs_mod = [v % modulus for v in scaled_inputs]
        expected_outputs_mod = [v % modulus for v in flat_outputs]

        actual_inputs = witness[:n_in]
        actual_outputs = witness[n_in : n_in + n_out]

        return compare_field_values(
            expected_inputs_mod,
            actual_inputs,
            modulus,
        ) and compare_field_values(expected_outputs_mod, actual_outputs, modulus)


# -------------------------
# Factory
# -------------------------
def get_loader(system: ZKProofSystems, path: str) -> WitnessLoader:
    if system == ZKProofSystems.Expander:
        return ExpanderWitnessLoader(path)
    msg = f"No loader implemented for {system}"
    raise ProofSystemNotImplementedError(msg)


# -------------------------
# Public API
# -------------------------
def load_witness(path: str, system: ZKProofSystems = ZKProofSystems.Expander) -> dict:
    loader = get_loader(system, path)
    return loader.load_witness()


def compare_witness_to_io(  # noqa: PLR0913
    witnesses: dict,
    expected_inputs: dict,
    expected_outputs: dict,
    modulus: int,
    system: ZKProofSystems = ZKProofSystems.Expander,
    scaling_function: Callable[[list[int], int, int], list[int]] | None = None,
) -> bool:
    """
    Compare witness data to expected inputs and outputs for a given ZK proof system.

    Args:
        witnesses (dict): Witness data as returned by `load_witness`.
        expected_inputs (dict):
            Dictionary containing key "input" mapping to list of expected integers.
        expected_outputs (dict):
            Dictionary containing key "output" mapping to list of expected integers.
        modulus (int): Field modulus.
        system (ZKProofSystems, optional):
            The ZK proof system. Defaults to ZKProofSystems.Expander.
        scaling_function (Callable[[list[int], int, int], list[int]] | None, optional):
            Optional scaling function to apply to inputs.
            Takes (inputs, scale_base, scale_exponent) and returns scaled inputs.
            Defaults to None.

    Returns:
        bool: True if the witness matches the expected I/O, False otherwise.
    """
    loader = get_loader(system, "")  # path not needed for comparison
    return loader.compare_witness_to_io(
        witnesses,
        expected_inputs,
        expected_outputs,
        modulus,
        scaling_function,
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    w = load_witness("./artifacts/lenet/witness.bin", ZKProofSystems.Expander)
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
