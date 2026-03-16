#include <metal_stdlib>
using namespace metal;

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

#define CIOS_ITER(AI, T0, T1, T2, T3, T4) \
{ \
    uint64_t carry; \
    uint64_t lo, hi, s, c1, s2, c2; \
    \
    lo = AI * b0; hi = mulhi(AI, b0); \
    s = lo + T0; c1 = (s < lo) ? 1ULL : 0ULL; \
    T0 = s; carry = hi + c1; \
    \
    lo = AI * b1; hi = mulhi(AI, b1); \
    s = lo + T1; c1 = (s < lo) ? 1ULL : 0ULL; \
    s2 = s + carry; c2 = (s2 < s) ? 1ULL : 0ULL; \
    T1 = s2; carry = hi + c1 + c2; \
    \
    lo = AI * b2; hi = mulhi(AI, b2); \
    s = lo + T2; c1 = (s < lo) ? 1ULL : 0ULL; \
    s2 = s + carry; c2 = (s2 < s) ? 1ULL : 0ULL; \
    T2 = s2; carry = hi + c1 + c2; \
    \
    lo = AI * b3; hi = mulhi(AI, b3); \
    s = lo + T3; c1 = (s < lo) ? 1ULL : 0ULL; \
    s2 = s + carry; c2 = (s2 < s) ? 1ULL : 0ULL; \
    T3 = s2; carry = hi + c1 + c2; \
    \
    T4 = T4 + carry; \
    \
    uint64_t m = T0 * BN254_INV; \
    \
    lo = m * BN254_MOD[0]; hi = mulhi(m, BN254_MOD[0]); \
    s = lo + T0; carry = hi + ((s < lo) ? 1ULL : 0ULL); \
    \
    lo = m * BN254_MOD[1]; hi = mulhi(m, BN254_MOD[1]); \
    s = lo + T1; c1 = (s < lo) ? 1ULL : 0ULL; \
    s2 = s + carry; c2 = (s2 < s) ? 1ULL : 0ULL; \
    T0 = s2; carry = hi + c1 + c2; \
    \
    lo = m * BN254_MOD[2]; hi = mulhi(m, BN254_MOD[2]); \
    s = lo + T2; c1 = (s < lo) ? 1ULL : 0ULL; \
    s2 = s + carry; c2 = (s2 < s) ? 1ULL : 0ULL; \
    T1 = s2; carry = hi + c1 + c2; \
    \
    lo = m * BN254_MOD[3]; hi = mulhi(m, BN254_MOD[3]); \
    s = lo + T3; c1 = (s < lo) ? 1ULL : 0ULL; \
    s2 = s + carry; c2 = (s2 < s) ? 1ULL : 0ULL; \
    T2 = s2; carry = hi + c1 + c2; \
    \
    s = T4 + carry; \
    uint64_t cc = (s < T4) ? 1ULL : 0ULL; \
    T3 = s; T4 = cc; \
}

inline BN254Fr bn254_mul(BN254Fr a, BN254Fr b) {
    uint64_t b0 = b.v[0], b1 = b.v[1], b2 = b.v[2], b3 = b.v[3];
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    CIOS_ITER(a.v[0], t0, t1, t2, t3, t4)
    CIOS_ITER(a.v[1], t0, t1, t2, t3, t4)
    CIOS_ITER(a.v[2], t0, t1, t2, t3, t4)
    CIOS_ITER(a.v[3], t0, t1, t2, t3, t4)

    if (t4 != 0 || bn254_gte_mod(t0, t1, t2, t3)) {
        uint64_t s0 = t0 - BN254_MOD[0];
        uint64_t bw0 = (t0 < BN254_MOD[0]) ? 1ULL : 0ULL;
        uint64_t s1 = t1 - BN254_MOD[1] - bw0;
        uint64_t bw1 = (bw0 ? (t1 <= BN254_MOD[1]) : (t1 < BN254_MOD[1])) ? 1ULL : 0ULL;
        uint64_t s2 = t2 - BN254_MOD[2] - bw1;
        uint64_t bw2 = (bw1 ? (t2 <= BN254_MOD[2]) : (t2 < BN254_MOD[2])) ? 1ULL : 0ULL;
        uint64_t s3 = t3 - BN254_MOD[3] - bw2;
        return BN254Fr{{s0, s1, s2, s3}};
    }
    return BN254Fr{{t0, t1, t2, t3}};
}
