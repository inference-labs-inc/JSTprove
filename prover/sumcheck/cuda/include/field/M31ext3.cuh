#pragma once

#include "basefield.cuh"

namespace gkr::M31ext3_field {
    // Mod of M31
    const int mod = 2147483647;
    #define mod_reduce_int(x) (x = (((x) & mod) + ((x) >> 31)))

    class M31ext3 final : public BaseField<M31ext3> {
    private:
        __host__ __device__
        static inline uint32_t mul_m31(const uint32_t& a, const uint32_t& b){
            uint64_t xx = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
            mod_reduce_int(xx);
            if (xx >= mod) xx -= mod;
            return xx;
        }

        __host__ __device__
        static inline uint32_t add_m31(const uint32_t& a, const uint32_t& b){
            uint32_t res;
            res = a + b;
            if(res >= mod) res -= mod;
            return res;
        }
    public:
        // internal storage of M31 extension-3
        uint32_t fs[3] = {0, 0, 0};

        static M31ext3 INV_2;

        __host__ __device__
        static M31ext3 zero() {
            M31ext3 f;
            f.fs[0] = 0;
            f.fs[1] = 0;
            f.fs[2] = 0;
            return f;
        }

        __host__ __device__
        static M31ext3 one() {
            M31ext3 f;
            f.fs[0] = 1;
            f.fs[1] = 0;
            f.fs[2] = 0;
            return f;
        }

        __host__
        static M31ext3 random(){
            M31ext3 f;
            f.fs[0] = static_cast<uint32_t>(rand());
            f.fs[1] = static_cast<uint32_t>(rand());
            f.fs[2] = static_cast<uint32_t>(rand());
            mod_reduce_int(f.fs[0]);
            mod_reduce_int(f.fs[1]);
            mod_reduce_int(f.fs[2]);
            return f;
        }

        __host__ __device__
        static inline M31ext3 new_unchecked(uint32_t x){
            M31ext3 f;
            f.fs[0] = x;
            return f;
        }

        __host__ __device__
        M31ext3() {
            this->fs[0] = 0;
            this->fs[1] = 0;
            this->fs[2] = 0;
        }

        __host__ __device__
        M31ext3(uint32_t v){
            mod_reduce_int(v);
            this->fs[0] = v;
        }

        __host__ __device__
        inline M31ext3 operator+(const M31ext3 &rhs) const{
            M31ext3 result;
            for(int i = 0; i < 3; i++){
                result.fs[i] = (fs[i] + rhs.fs[i]);
                if (result.fs[i] >= mod) result.fs[i] -= mod;
            }
            return result;
        }

        __host__ __device__
        inline M31ext3 operator*(const M31ext3 &rhs) const{
            M31ext3 f;

            //            let a = &a.v;
            //            let b = &b.v;
            //            let mut res = [M31::default(); 3];
            //            res[0] = a[0] * b[0] + M31 { v: 5 } * (a[1] * b[2] + a[2] * b[1]);
            //            res[1] = a[0] * b[1] + a[1] * b[0] + M31 { v: 5 } * a[2] * b[2];
            //            res[2] = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];

             f.fs[0] = add_m31(
                     mul_m31(fs[0], rhs.fs[0]),
                     mul_m31(5, add_m31(
                             mul_m31(fs[1], rhs.fs[2]),
                             mul_m31(fs[2], rhs.fs[1]))));

             f.fs[1] = add_m31(
                    add_m31(
                            mul_m31(fs[0], rhs.fs[1]),
                            mul_m31(fs[1], rhs.fs[0])
                    ),
                    mul_m31(5, mul_m31(fs[2], rhs.fs[2]))
            );

            f.fs[2] = add_m31(
                    add_m31(
                            mul_m31(fs[0], rhs.fs[2]),
                            mul_m31(fs[1], rhs.fs[1])
                    ),
                    mul_m31(fs[2], rhs.fs[0])
            );

            return f;
        }

        __host__ __device__
        inline M31ext3 operator-() const{
            M31ext3 f;
            for(int i = 0; i < 3; i++) {
                f.fs[i] = (this->fs[i] == 0) ? 0 : (mod - this->fs[i]);
            }
            return f;
        }

        __host__ __device__
        inline M31ext3 operator-(const M31ext3 &rhs) const{
            return *this + (-rhs);
        }

        __host__ __device__
        bool operator==(const M31ext3 &rhs) const{
            return (this->fs[0] == rhs.fs[0]) && (this->fs[1] == rhs.fs[1]) && (this->fs[2] == rhs.fs[2]);
        };

        // From field to transcript as bytes
        void to_bytes(uint8_t* output) const{
            memcpy(output, this, sizeof(*this));
        };

        // Convert from transcript bytes to Field
        void from_bytes(const uint8_t* input){
            memcpy(this, input, 12);
            for(int i = 0; i < 3; i++) {
                mod_reduce_int(fs[i]);
                if (fs[i] >= mod) fs[i] -= mod;
            }
        };
    };
    M31ext3 M31ext3::INV_2 = (1 << 30);
}
