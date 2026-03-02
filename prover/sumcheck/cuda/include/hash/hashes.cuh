#pragma once

#include "sha256.cuh"

namespace gkr{
    // Trait of Hasher
    class Hasher{
    public:
        virtual void hash(uint8_t *output, const uint8_t *input, uint32_t input_len){}
    };

    // SHA2-256 Hasher for Fiat-shamir
    class SHA256Hasher: public Hasher{
    public:
        CSHA256 btc_sha256_hasher;
        void hash(uint8_t *output, const uint8_t *input, uint32_t input_len) override {
            btc_sha256_hasher.Reset().Write(input, input_len).Finalize(output);
        }
    };
}
