// Gate accumulation intentionally kept on CPU.
// Metal device memory does not support spinlocks or acquire/release
// memory ordering on atomics, which are required for safe concurrent
// BN254 field element accumulation (256-bit locked add).
