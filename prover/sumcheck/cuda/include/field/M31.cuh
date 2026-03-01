#pragma once

#include "basefield.cuh"

namespace gkr::M31_field {

    const int mod = 2147483647;
    #define mod_reduce_int(x) (x = (((x) & mod) + ((x) >> 31)))

    class M31 final : public BaseField<M31> {
    public:
        // internal storage of M31
        uint32_t x;

        static M31 INV_2;

        __host__ __device__
        static M31 zero() { return new_unchecked(0); }

        __host__ __device__
        static M31 one() { return new_unchecked(1); }

        __host__
        static M31 random() {
            return M31{static_cast<uint32_t>(rand())};
        }

        __host__ __device__
        static inline M31 new_unchecked(uint32_t x){
            M31 f;
            f.x = x;
            return f;
        }

        __host__ __device__
        M31() { this->x = 0; }

        __host__ __device__
        M31(uint32_t v){
            mod_reduce_int(v);
            this->x = v;
        }

        __host__ __device__
        inline M31 operator+(const M31 &rhs) const{
            M31 result;
            result.x = (x + rhs.x);
            if (result.x >= mod) result.x -= mod;
            return result;
        }

        __host__ __device__
        inline M31 operator*(const M31 &rhs) const{
            int64_t xx = static_cast<int64_t>(x) * rhs.x;
            mod_reduce_int(xx);
            if (xx >= mod) xx -= mod;
            return new_unchecked(xx);
        }

        __host__ __device__
        inline M31 operator-() const{
            uint32_t xx = (this->x == 0) ? 0 : (mod - this->x);
            return new_unchecked(xx);
        }

        __host__ __device__
        inline M31 operator-(const M31 &rhs) const{
            return *this + (-rhs);
        }

        __host__ __device__
        bool operator==(const M31 &rhs) const{
            return this->x == rhs.x;
        };

        // From field to transcript as bytes
        void to_bytes(uint8_t* output) const{
            memcpy(output, this, sizeof(*this));
        };

        // Convert from transcript bytes to Field
        void from_bytes(const uint8_t* input){
            memcpy(this, input, 4);
            mod_reduce_int(x);
            if (x >= mod) x -= mod;
        };
    };

    M31 M31::INV_2 = (1 << 30);
}
