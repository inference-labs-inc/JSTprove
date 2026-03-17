#include <metal_stdlib>
using namespace metal;

// BN254 scalar field Fr
// r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
// Internal representation: [u64; 4] in Montgomery form (little-endian limbs)
// Fr(a) = a * R mod r, where R = 2^256

constant uint64_t BN254_MOD[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

constant uint64_t BN254_INV = 0xc2e1f593efffffffULL;

struct BN254Fr {
    uint64_t v[4];
};

inline BN254Fr bn254_zero() {
    return BN254Fr{{0, 0, 0, 0}};
}

inline bool bn254_gte_mod(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    if (a3 != BN254_MOD[3]) return a3 > BN254_MOD[3];
    if (a2 != BN254_MOD[2]) return a2 > BN254_MOD[2];
    if (a1 != BN254_MOD[1]) return a1 > BN254_MOD[1];
    return a0 >= BN254_MOD[0];
}

inline BN254Fr bn254_add(BN254Fr a, BN254Fr b) {
    uint64_t r0 = a.v[0] + b.v[0];
    uint64_t c0 = (r0 < a.v[0]) ? 1ULL : 0ULL;

    uint64_t r1 = a.v[1] + b.v[1] + c0;
    uint64_t c1 = (c0 ? (r1 <= a.v[1]) : (r1 < a.v[1])) ? 1ULL : 0ULL;

    uint64_t r2 = a.v[2] + b.v[2] + c1;
    uint64_t c2 = (c1 ? (r2 <= a.v[2]) : (r2 < a.v[2])) ? 1ULL : 0ULL;

    uint64_t r3 = a.v[3] + b.v[3] + c2;

    if (bn254_gte_mod(r0, r1, r2, r3)) {
        uint64_t s0 = r0 - BN254_MOD[0];
        uint64_t b0 = (r0 < BN254_MOD[0]) ? 1ULL : 0ULL;
        uint64_t s1 = r1 - BN254_MOD[1] - b0;
        uint64_t b1 = (b0 ? (r1 <= BN254_MOD[1]) : (r1 < BN254_MOD[1])) ? 1ULL : 0ULL;
        uint64_t s2 = r2 - BN254_MOD[2] - b1;
        uint64_t b2 = (b1 ? (r2 <= BN254_MOD[2]) : (r2 < BN254_MOD[2])) ? 1ULL : 0ULL;
        uint64_t s3 = r3 - BN254_MOD[3] - b2;
        return BN254Fr{{s0, s1, s2, s3}};
    }
    return BN254Fr{{r0, r1, r2, r3}};
}

inline BN254Fr bn254_sub(BN254Fr a, BN254Fr b) {
    uint64_t r0 = a.v[0] - b.v[0];
    uint64_t bw0 = (a.v[0] < b.v[0]) ? 1ULL : 0ULL;

    uint64_t r1 = a.v[1] - b.v[1] - bw0;
    uint64_t bw1 = (bw0 ? (a.v[1] <= b.v[1]) : (a.v[1] < b.v[1])) ? 1ULL : 0ULL;

    uint64_t r2 = a.v[2] - b.v[2] - bw1;
    uint64_t bw2 = (bw1 ? (a.v[2] <= b.v[2]) : (a.v[2] < b.v[2])) ? 1ULL : 0ULL;

    uint64_t r3 = a.v[3] - b.v[3] - bw2;
    uint64_t bw3 = (bw2 ? (a.v[3] <= b.v[3]) : (a.v[3] < b.v[3])) ? 1ULL : 0ULL;

    if (bw3) {
        uint64_t s0 = r0 + BN254_MOD[0];
        uint64_t c0 = (s0 < r0) ? 1ULL : 0ULL;
        uint64_t s1 = r1 + BN254_MOD[1] + c0;
        uint64_t c1 = (c0 ? (s1 <= r1) : (s1 < r1)) ? 1ULL : 0ULL;
        uint64_t s2 = r2 + BN254_MOD[2] + c1;
        uint64_t c2 = (c1 ? (s2 <= r2) : (s2 < r2)) ? 1ULL : 0ULL;
        uint64_t s3 = r3 + BN254_MOD[3] + c2;
        return BN254Fr{{s0, s1, s2, s3}};
    }
    return BN254Fr{{r0, r1, r2, r3}};
}

// Montgomery multiplication using CIOS
// a, b in Montgomery form → result in Montgomery form
inline BN254Fr bn254_mul(BN254Fr a, BN254Fr b) {
    uint64_t t[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        // Multiply: t += a[i] * b
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo = a.v[i] * b.v[j];
            uint64_t hi = mulhi(a.v[i], b.v[j]);
            uint64_t s = lo + t[j];
            uint64_t c1 = (s < lo) ? 1ULL : 0ULL;
            uint64_t s2 = s + carry;
            uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
            t[j] = s2;
            carry = hi + c1 + c2;
        }
        uint64_t s4 = t[4] + carry;
        t[4] = s4;

        // Reduce: m = t[0] * INV mod 2^64
        uint64_t m = t[0] * BN254_INV;

        // t += m * MODULUS, then shift right by 64
        {
            uint64_t lo = m * BN254_MOD[0];
            uint64_t hi = mulhi(m, BN254_MOD[0]);
            uint64_t s = lo + t[0];
            carry = hi + ((s < lo) ? 1ULL : 0ULL);
        }
        for (int j = 1; j < 4; j++) {
            uint64_t lo = m * BN254_MOD[j];
            uint64_t hi = mulhi(m, BN254_MOD[j]);
            uint64_t s = lo + t[j];
            uint64_t c1 = (s < lo) ? 1ULL : 0ULL;
            uint64_t s2 = s + carry;
            uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
            t[j - 1] = s2;
            carry = hi + c1 + c2;
        }
        uint64_t s = t[4] + carry;
        uint64_t c = (s < t[4]) ? 1ULL : 0ULL;
        t[3] = s;
        t[4] = c;
    }

    // Final reduction
    if (t[4] != 0 || bn254_gte_mod(t[0], t[1], t[2], t[3])) {
        uint64_t s0 = t[0] - BN254_MOD[0];
        uint64_t bw0 = (t[0] < BN254_MOD[0]) ? 1ULL : 0ULL;
        uint64_t s1 = t[1] - BN254_MOD[1] - bw0;
        uint64_t bw1 = (bw0 ? (t[1] <= BN254_MOD[1]) : (t[1] < BN254_MOD[1])) ? 1ULL : 0ULL;
        uint64_t s2 = t[2] - BN254_MOD[2] - bw1;
        uint64_t bw2 = (bw1 ? (t[2] <= BN254_MOD[2]) : (t[2] < BN254_MOD[2])) ? 1ULL : 0ULL;
        uint64_t s3 = t[3] - BN254_MOD[3] - bw2;
        return BN254Fr{{s0, s1, s2, s3}};
    }
    return BN254Fr{{t[0], t[1], t[2], t[3]}};
}

inline BN254Fr bn254_simd_shuffle_down(BN254Fr val, ushort offset) {
    for (int i = 0; i < 4; i++) {
        uint32_t lo = (uint32_t)(val.v[i] & 0xFFFFFFFF);
        uint32_t hi = (uint32_t)(val.v[i] >> 32);
        lo = simd_shuffle_down(lo, offset);
        hi = simd_shuffle_down(hi, offset);
        val.v[i] = ((uint64_t)hi << 32) | (uint64_t)lo;
    }
    return val;
}

inline BN254Fr bn254_simd_sum(BN254Fr val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val = bn254_add(val, bn254_simd_shuffle_down(val, offset));
    }
    return val;
}
