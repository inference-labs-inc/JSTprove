#pragma once

#include "hash/hashes.cuh"

#define DIGEST_SIZE         32
#define MAX_PROOF_BYTE_SIZE 8192

namespace gkr{

    template<typename F>
    class Proof{
    public:
        uint32_t idx, bytes_write_ptr, commitment_nb_bytes, opening_nb_bytes;
        uint8_t bytes[MAX_PROOF_BYTE_SIZE];

        Proof(){
            idx = 0;
            bytes_write_ptr = 0;
        }

        void append_bytes(const uint8_t* __restrict__ input_bytes, uint32_t len){
            for(int l_idx = 0; l_idx < len; l_idx++){
                bytes[bytes_write_ptr + l_idx] = input_bytes[l_idx];
            }
            bytes_write_ptr += len;
            assert(bytes_write_ptr < MAX_PROOF_BYTE_SIZE);
        }

        inline void reset(){
            idx = 0;
            bytes_write_ptr = 0;
        }

        const uint8_t* bytes_head(){
            return bytes + idx;
        }

        inline void step(uint32_t nb_bytes){
            idx += nb_bytes;
        }

        inline F get_next_and_step(){
            F f;
            f.from_bytes(bytes + idx);
            step(sizeof(F));
            return f;
        }
    };

    template <typename F, typename F_primitive>
    class Transcript{
    private:
        inline void _hash_to_digest(){
            uint32_t hash_end_idx = proof.bytes_write_ptr;
            if (hash_end_idx - hash_start_idx > 0)
            {
                hasher.hash(digest, proof.bytes + hash_start_idx, hash_end_idx - hash_start_idx);
                hash_start_idx = hash_end_idx;
            }
            else
            {
                hasher.hash(digest, digest, DIGEST_SIZE);
            }
        }

    public:
        Proof<F> proof;
        SHA256Hasher hasher;
        uint32_t hash_start_idx;
        uint8_t digest[DIGEST_SIZE];

        Transcript(){
            proof = Proof<F>();
            hasher = SHA256Hasher();
            hash_start_idx = 0;
            for(unsigned char & i : digest){ i = 0; }
        }

        void append_bytes(const uint8_t* bytes, uint32_t len){
            proof.append_bytes(bytes, len);
        }

        void append_f(const F &f){
            uint32_t cur_size = proof.bytes_write_ptr;
            proof.bytes_write_ptr += sizeof(F);
            f.to_bytes(proof.bytes + cur_size);
        }

        F_primitive challenge_f(){
            _hash_to_digest();

            F_primitive f;
            assert(sizeof(F_primitive) <= DIGEST_SIZE);
            f.from_bytes(digest);
            return f;
        }
    };

}
