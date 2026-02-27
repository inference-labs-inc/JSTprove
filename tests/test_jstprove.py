"""Python API tests for jstprove bindings.

These tests exercise the public surface of the jstprove Python package
without requiring compiled model files. They cover:

- Import and type correctness
- Constructor behaviour
- Error propagation from Rust → Python
- Thread safety and GIL-release correctness
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from jstprove import BatchResult, Circuit, WitnessResult


# ── Import and type surface ───────────────────────────────────────────────────


def test_circuit_class_importable():
    assert Circuit is not None


def test_witness_result_class_importable():
    assert WitnessResult is not None


def test_batch_result_class_importable():
    assert BatchResult is not None


def test_circuit_is_type():
    assert isinstance(Circuit, type)


def test_witness_result_is_type():
    assert isinstance(WitnessResult, type)


def test_batch_result_is_type():
    assert isinstance(BatchResult, type)


# ── Constructor behaviour ─────────────────────────────────────────────────────


def test_circuit_constructor_accepts_arbitrary_path():
    c = Circuit("/any/path/to/model.bundle")
    assert isinstance(c, Circuit)


def test_circuit_constructor_accepts_empty_string():
    c = Circuit("")
    assert isinstance(c, Circuit)


def test_witness_result_not_user_constructable():
    with pytest.raises(TypeError):
        WitnessResult()


def test_batch_result_not_user_constructable():
    with pytest.raises(TypeError):
        BatchResult()


# ── API surface: methods and attributes ──────────────────────────────────────


@pytest.mark.parametrize(
    "method",
    [
        "generate_witness",
        "prove",
        "verify",
        "generate_witness_batch",
        "prove_batch",
        "verify_batch",
    ],
)
def test_circuit_has_instance_method(method):
    assert callable(getattr(Circuit, method))


@pytest.mark.parametrize("method", ["compile", "is_compatible"])
def test_circuit_has_static_method(method):
    assert callable(getattr(Circuit, method))


def test_witness_result_exposes_witness_path():
    assert hasattr(WitnessResult, "witness_path")


def test_witness_result_exposes_output_path():
    assert hasattr(WitnessResult, "output_path")


def test_batch_result_exposes_succeeded():
    assert hasattr(BatchResult, "succeeded")


def test_batch_result_exposes_failed():
    assert hasattr(BatchResult, "failed")


def test_batch_result_exposes_errors():
    assert hasattr(BatchResult, "errors")


# ── is_compatible with invalid / absent paths ─────────────────────────────────


def test_is_compatible_nonexistent_file_returns_false():
    compatible, _ = Circuit.is_compatible("/does/not/exist/model.onnx")
    assert compatible is False


def test_is_compatible_nonexistent_file_returns_issues_list():
    _, issues = Circuit.is_compatible("/does/not/exist/model.onnx")
    assert isinstance(issues, list)
    assert len(issues) > 0
    assert isinstance(issues[0], str)


def test_is_compatible_returns_two_tuple():
    result = Circuit.is_compatible("/nonexistent.onnx")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_is_compatible_first_element_is_bool():
    compatible, _ = Circuit.is_compatible("/nonexistent.onnx")
    assert isinstance(compatible, bool)


# ── Error propagation: all operations raise RuntimeError on bad paths ─────────


def test_generate_witness_bad_paths_raises_runtime_error(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError):
        circuit.generate_witness(
            str(tmp_path / "input.msgpack"),
            str(tmp_path / "witness.msgpack"),
        )


def test_prove_bad_paths_raises_runtime_error(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError):
        circuit.prove(
            str(tmp_path / "witness.msgpack"),
            str(tmp_path / "proof.msgpack"),
        )


def test_verify_bad_paths_raises_runtime_error(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError):
        circuit.verify(
            str(tmp_path / "proof.msgpack"),
            str(tmp_path / "input.msgpack"),
        )


def test_generate_witness_batch_bad_manifest_raises(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError):
        circuit.generate_witness_batch(str(tmp_path / "manifest.json"))


def test_prove_batch_bad_manifest_raises(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError):
        circuit.prove_batch(str(tmp_path / "manifest.json"))


def test_verify_batch_bad_manifest_raises(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError):
        circuit.verify_batch(str(tmp_path / "manifest.json"))


def test_compile_bad_onnx_path_raises_runtime_error(tmp_path):
    with pytest.raises(RuntimeError):
        Circuit.compile(
            str(tmp_path / "nonexistent.onnx"),
            str(tmp_path / "output.bundle"),
        )


def test_error_type_is_runtime_error_not_base_exception(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    exc = None
    try:
        circuit.generate_witness(str(tmp_path / "a"), str(tmp_path / "b"))
    except RuntimeError as e:
        exc = e
    assert exc is not None
    assert type(exc).__name__ == "RuntimeError"


def test_error_message_is_non_empty_string(tmp_path):
    circuit = Circuit(str(tmp_path / "no_model.bundle"))
    with pytest.raises(RuntimeError, match=r".+"):
        circuit.generate_witness(str(tmp_path / "a"), str(tmp_path / "b"))


# ── Thread safety and GIL-release ─────────────────────────────────────────────


def test_concurrent_generate_witness_calls_complete_without_deadlock(tmp_path):
    """Two threads calling generate_witness concurrently must both complete.

    Without py.allow_threads in the Rust binding, the second thread would
    block on the GIL for the entire duration of the first call. With
    allow_threads the GIL is released so both execute in parallel.
    Either way they should complete; this guards against deadlock.
    """
    circuit = Circuit(str(tmp_path / "model.bundle"))
    errors: list[BaseException | None] = [None, None]

    def call(idx: int) -> None:
        try:
            circuit.generate_witness(
                str(tmp_path / f"in_{idx}.msgpack"),
                str(tmp_path / f"out_{idx}.msgpack"),
            )
        except RuntimeError as e:
            errors[idx] = e

    t1 = threading.Thread(target=call, args=(0,))
    t2 = threading.Thread(target=call, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not t1.is_alive(), "thread 1 did not complete — possible GIL deadlock"
    assert not t2.is_alive(), "thread 2 did not complete — possible GIL deadlock"
    assert isinstance(errors[0], RuntimeError), "thread 1 should have raised RuntimeError"
    assert isinstance(errors[1], RuntimeError), "thread 2 should have raised RuntimeError"


def test_concurrent_is_compatible_calls_return_independent_results(tmp_path):
    """N threads calling is_compatible should each receive correct results
    with no shared-state corruption between threads."""
    N = 8
    results: list[tuple[bool, list[str]] | None] = [None] * N

    def check(idx: int) -> None:
        results[idx] = Circuit.is_compatible(str(tmp_path / f"model_{idx}.onnx"))

    threads = [threading.Thread(target=check, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    for idx, result in enumerate(results):
        assert result is not None, f"thread {idx} produced no result"
        compatible, issues = result
        assert compatible is False
        assert isinstance(issues, list)


def test_thread_pool_executor_dispatches_all_workers(tmp_path):
    """ThreadPoolExecutor with N workers must complete all tasks.

    With the GIL held for each Rust call, workers would execute
    strictly serially. With allow_threads they overlap. In both cases
    all tasks should complete (no starvation, no deadlock).
    """
    circuit = Circuit(str(tmp_path / "model.bundle"))
    N = 4

    def worker(idx: int) -> RuntimeError | None:
        try:
            circuit.generate_witness(
                str(tmp_path / f"in_{idx}.msgpack"),
                str(tmp_path / f"out_{idx}.msgpack"),
            )
            return None
        except RuntimeError as e:
            return e

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(worker, i) for i in range(N)]
        results = [f.result(timeout=30) for f in as_completed(futures)]

    assert len(results) == N
    assert all(isinstance(r, RuntimeError) for r in results), (
        "all workers should have received RuntimeError for nonexistent paths"
    )


def test_python_thread_not_starved_during_rust_calls(tmp_path):
    """Python background threads must make progress while Rust executes.

    A Python counter thread is started first. We then issue repeated
    Rust calls (which fail fast on missing files). If the GIL is held
    throughout each call AND between calls, the counter never advances.

    Note: this test is a smoke-test for complete starvation. A definitive
    single-call GIL test requires a longer-running operation (e.g. actual
    witness generation on a real model). Run the full e2e suite to stress
    the allow_threads paths under real workloads.
    """
    circuit = Circuit(str(tmp_path / "model.bundle"))
    counter = [0]
    stop = threading.Event()

    def count() -> None:
        while not stop.is_set():
            counter[0] += 1

    t = threading.Thread(target=count, daemon=True)
    t.start()
    time.sleep(0.01)

    for _ in range(50):
        try:
            circuit.generate_witness(
                str(tmp_path / "in.msgpack"),
                str(tmp_path / "out.msgpack"),
            )
        except RuntimeError:
            pass

    stop.set()
    t.join(timeout=2.0)

    assert counter[0] > 0, (
        "background Python thread never advanced — GIL was not released "
        "between Rust calls"
    )
