#pragma once

#include <cstdint>

namespace gkr{

    template <typename F>
    class BaseField{
    public:
        __host__ __device__
        static F zero();

        __host__ __device__
        static F one();

        __host__ __device__
        static F random();

        __host__ __device__
        F operator+(const F &rhs) const;

        __host__ __device__
        F operator*(const F &rhs) const;

        __host__ __device__
        F operator-() const;

        __host__ __device__
        F operator-(const F &rhs) const;

        __host__ __device__
        bool operator==(const F &rhs) const;

        __host__ __device__
        bool operator!=(const F&rhs) { return !(*this == rhs);}

        __host__ __device__
        void operator+=(const F &rhs) { *static_cast<F *>(this) = *static_cast<F *>(this) + rhs; }

        __host__ __device__
        void operator-=(const F &rhs) { *static_cast<F *>(this) = *static_cast<F *>(this) - rhs; }

        __host__ __device__
        void operator*=(const F &rhs) { *static_cast<F *>(this) = *static_cast<F *>(this) * rhs; }

        // Host Function that runs on CPU
        void to_bytes(uint8_t *output) const;
        void from_bytes(const uint8_t *input);
    };
}
