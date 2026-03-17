#include <metal_stdlib>
using namespace metal;

constant uint64_t GL_MOD = 0xFFFFFFFF00000001ULL;
constant uint64_t GL_EPSILON = 0xFFFFFFFFULL;

struct GoldilocksFr {
    uint64_t v;
};

inline GoldilocksFr gl_zero() {
    return GoldilocksFr{0};
}

inline uint64_t gl_reduce(uint64_t x) {
    return (x >= GL_MOD) ? (x - GL_MOD) : x;
}

inline GoldilocksFr gl_add(GoldilocksFr a, GoldilocksFr b) {
    uint64_t sum = a.v + b.v;
    uint64_t carry = (sum < a.v) ? 1ULL : 0ULL;
    if (carry) {
        sum += GL_EPSILON;
        sum = gl_reduce(sum);
    } else {
        sum = gl_reduce(sum);
    }
    return GoldilocksFr{sum};
}

inline GoldilocksFr gl_sub(GoldilocksFr a, GoldilocksFr b) {
    if (a.v >= b.v) {
        return GoldilocksFr{a.v - b.v};
    }
    return GoldilocksFr{a.v + GL_MOD - b.v};
}

inline GoldilocksFr gl_mul(GoldilocksFr a, GoldilocksFr b) {
    uint64_t lo = a.v * b.v;
    uint64_t hi = mulhi(a.v, b.v);

    uint64_t hi_lo = hi & GL_EPSILON;
    uint64_t hi_hi = hi >> 32;

    uint64_t t0 = lo - hi_hi;
    uint64_t borrow0 = (lo < hi_hi) ? 1ULL : 0ULL;

    uint64_t t1 = hi_lo * GL_EPSILON;
    uint64_t t1_hi = mulhi(hi_lo, GL_EPSILON);

    uint64_t r = t0 - t1;
    uint64_t borrow1 = (t0 < t1) ? 1ULL : 0ULL;

    uint64_t total_borrow = borrow0 + borrow1 + t1_hi;
    r += total_borrow * GL_MOD;

    r = gl_reduce(r);
    r = gl_reduce(r);

    return GoldilocksFr{r};
}
