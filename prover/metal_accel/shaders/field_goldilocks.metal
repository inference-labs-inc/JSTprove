#include <metal_stdlib>
using namespace metal;

constant uint64_t GL_P = 0xFFFFFFFF00000001ULL;

struct GoldilocksField {
    uint64_t val;
};

inline GoldilocksField gl_zero() {
    return GoldilocksField{0};
}

inline GoldilocksField gl_one() {
    return GoldilocksField{1};
}

inline GoldilocksField gl_reduce(uint64_t x) {
    uint64_t r = x >= GL_P ? x - GL_P : x;
    return GoldilocksField{r};
}

inline GoldilocksField gl_add(GoldilocksField a, GoldilocksField b) {
    uint64_t sum = a.val + b.val;
    uint64_t r = sum - GL_P;
    // branch-free: if sum < GL_P, the subtraction underflows and sum < r
    return GoldilocksField{sum < a.val || sum >= GL_P ? r : sum};
}

inline GoldilocksField gl_sub(GoldilocksField a, GoldilocksField b) {
    uint64_t r = a.val - b.val;
    // branch-free: if a < b, add p back
    return GoldilocksField{a.val >= b.val ? r : r + GL_P};
}

// Goldilocks reduction of a 128-bit product:
// Given product = hi:lo (128-bit), reduce mod p = 2^64 - 2^32 + 1
//
// Derived from: 2^64 ≡ 2^32 - 1 (mod p)
//
// We decompose hi into hi_lo (lower 32 bits) and hi_hi (upper 32 bits):
//   product = lo + hi * 2^64
//           ≡ lo + hi_lo * (2^32 - 1) - hi_hi  (mod p)
//           = lo + hi_lo * 2^32 - hi_lo - hi_hi
inline GoldilocksField gl_reduce128(uint64_t lo, uint64_t hi) {
    uint64_t hi_lo = hi & 0xFFFFFFFF;
    uint64_t hi_hi = hi >> 32;

    // Compute: lo + hi_lo * 2^32 - hi_lo - hi_hi
    // Split into: (lo - hi_lo - hi_hi) + (hi_lo << 32)
    // Handle underflow carefully

    uint64_t shifted = hi_lo << 32;

    // Accumulate with carries tracked via overflows
    // Step 1: lo + shifted
    uint64_t s1 = lo + shifted;
    uint64_t c1 = (s1 < lo) ? 1ULL : 0ULL;

    // Step 2: s1 - hi_lo
    uint64_t s2 = s1 - hi_lo;
    uint64_t c2 = (s1 < hi_lo) ? 1ULL : 0ULL;

    // Step 3: s2 - hi_hi
    uint64_t s3 = s2 - hi_hi;
    uint64_t c3 = (s2 < hi_hi) ? 1ULL : 0ULL;

    // Net carry: c1 - c2 - c3 can be -2, -1, 0, or 1
    // 1 * 2^64 ≡ 2^32 - 1 (mod p)
    // -1 * 2^64 ≡ -(2^32 - 1) ≡ p - 2^32 + 1 = 2^64 - 2^33 + 2 (mod p)
    int64_t carry = int64_t(c1) - int64_t(c2) - int64_t(c3);

    uint64_t result = s3;
    if (carry > 0) {
        // Add (2^32 - 1) per unit of positive carry
        uint64_t adj = uint64_t(carry) * 0xFFFFFFFFULL;
        result += adj;
        if (result < adj) result -= GL_P; // overflow wrap
    } else if (carry < 0) {
        // Subtract (2^32 - 1) per unit of negative carry
        uint64_t adj = uint64_t(-carry) * 0xFFFFFFFFULL;
        if (result >= adj) {
            result -= adj;
        } else {
            result = result + GL_P - adj;
            if (result >= GL_P) result -= GL_P;
        }
    }

    // Final reduction
    if (result >= GL_P) result -= GL_P;
    return GoldilocksField{result};
}

inline GoldilocksField gl_mul(GoldilocksField a, GoldilocksField b) {
    // 64x64 → 128-bit multiply using Metal's mulhi
    uint64_t lo = a.val * b.val;
    uint64_t hi = mulhi(a.val, b.val);
    return gl_reduce128(lo, hi);
}

inline GoldilocksField gl_scale(GoldilocksField a, GoldilocksField r) {
    return gl_mul(a, r);
}

inline GoldilocksField gl_neg(GoldilocksField a) {
    return a.val == 0 ? gl_zero() : GoldilocksField{GL_P - a.val};
}
