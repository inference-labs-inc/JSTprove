#pragma once

#include "basefield.cuh"

// Icicle Library
#ifdef useBN254
#include "fields/id.h"
#define CURVE_ID BN254
#define FIELD_ID BN254
#include "fields/field_config.cuh"
#include "curves/curve_config.cuh"
#include "gpu-utils/modifiers.cuh"

typedef curve_config::scalar_t  field_t;

namespace gkr::BN254_field {

    class Bn254 final : public BaseField<Bn254> {
    public:

        // internal
        field_t fd __attribute__((packed));

        static Bn254 INV_2;

        __host__ __device__
        static Bn254 zero() {
            Bn254 f;
            f.fd = field_t ::zero();
            return f;
        }

        __host__ __device__
        static Bn254 one() {
            Bn254 f;
            f.fd = field_t ::one();
            return f;
        }

        __host__
        static Bn254 random(){
            Bn254 f;
            f.fd = field_t ::rand_host();
            return f;
        }

        __host__ __device__
        Bn254() {
            this->fd = field_t ::zero();
        }

        __host__ __device__
        Bn254(uint32_t v){
            this->fd = field_t ::from(v);
        }

        __host__ __device__
        Bn254(field_t ff){
            this->fd = ff;
        }

        __host__ __device__
        inline Bn254 operator+(const Bn254 &rhs) const{
            Bn254 result;
            result.fd = fd + rhs.fd;
            return result;
        }

        __host__ __device__
        inline Bn254 operator*(const Bn254 &rhs) const{
            Bn254 result;
            result.fd = fd * rhs.fd;
            return result;
        }

        __host__ __device__
        inline Bn254 operator-() const{
            Bn254 res;
            res.fd = field_t::neg(fd);
            return res;
        }

        __host__ __device__
        inline Bn254 operator-(const Bn254 &rhs) const{
            Bn254 res;
            res.fd = fd - rhs.fd;
            return res;
        }

        __host__ __device__
        bool operator==(const Bn254 &rhs) const{
            return fd == rhs.fd;
        };

        // From field to transcript as bytes
        void to_bytes(uint8_t* output) const{
            memcpy(output, this, sizeof(*this));
        };

        // Convert from transcript bytes to Field
        void from_bytes(const uint8_t* input){
            memcpy(this, input, 32);
            while (field_t::lt( field_t{field_t::get_modulus()}, fd))
                fd = fd - field_t{field_t::get_modulus()};
        };
    };
    Bn254 Bn254::INV_2 = field_t ::inverse(field_t::from(2));
}
#endif
