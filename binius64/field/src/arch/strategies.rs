// Copyright 2024-2025 Irreducible Inc.

/// Packed strategy for arithmetic operations.
/// (Uses arithmetic operations with underlier and subfield to simultaneously calculate the result
/// for all packed values)
pub struct PackedStrategy;
/// Pairwise recursive strategy. Calculates the result by applying recursive algorithm for each
/// packed value independently.
pub struct PairwiseRecursiveStrategy;
/// Pairwise strategy. Apply the result of the operation to each packed element independently.
pub struct PairwiseStrategy;
/// Get result of operation from the table for each sub-element
pub struct PairwiseTableStrategy;
/// Applicable only for multiply by alpha and square operations.
/// Reuse multiplication operation for that.
pub struct ReuseMultiplyStrategy;

/// Use operations with GFNI instructions
pub struct GfniStrategy;
/// Use SIMD operations for packed arithmetic
pub struct SimdStrategy;
/// Specialized versions of the above to resolve conflicting implementations
pub struct GfniSpecializedStrategy256b;
pub struct GfniSpecializedStrategy512b;

/// Strategy for BinaryField1b operations using bitwise AND for multiplication.
pub struct BitwiseAndStrategy;

/// Strategy for ScaledUnderlier operations that delegate to sub-underlier operations.
pub struct ScaledStrategy;
