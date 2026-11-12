from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import jstprove
from jstprove import holographic_setup_vk


def test_public_surface():
    assert jstprove.__all__ == ["holographic_setup_vk"]
    assert callable(holographic_setup_vk)


def test_unknown_proof_config_raises_value_error():
    with pytest.raises(ValueError, match="unknown proof_config"):
        holographic_setup_vk("not_a_config", b"")


@pytest.mark.parametrize(
    "config",
    [
        "bn254",
        "goldilocks_raw",
        "goldilocks_basefold",
        "goldilocks_ext2_basefold",
        "goldilocks_ext3_whir",
    ],
)
def test_non_holographic_config_raises_runtime_error(config):
    with pytest.raises(RuntimeError):
        holographic_setup_vk(config, b"")


def test_malformed_circuit_bytes_raise_runtime_error():
    with pytest.raises(RuntimeError):
        holographic_setup_vk("goldilocks_ext4_whir", b"\x00\x01\x02\x03")


def test_concurrent_calls_complete():
    def call(_):
        try:
            holographic_setup_vk("bn254", b"")
        except RuntimeError as e:
            return str(e)
        return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(32)))
    assert len(results) == 32
    assert all(results)
