use super::m31_utils::{
    big_array_add_no_reduce, bit_array_to_m31_26, bytes_to_bits, cap_sigma0, cap_sigma1, ch,
    m31_26_array_put_uint32, m31_26_to_bit_array_seperate, maj, sha_m31_26_add, sigma0, sigma1,
};
use expander_compiler::frontend::{Config, RootAPI, Variable};

const SHA256LEN: usize = 32;
const CHUNK: usize = 64;
const INIT0: u32 = 0x6A09E667;
const INIT1: u32 = 0xBB67AE85;
const INIT2: u32 = 0x3C6EF372;
const INIT3: u32 = 0xA54FF53A;
const INIT4: u32 = 0x510E527F;
const INIT5: u32 = 0x9B05688C;
const INIT6: u32 = 0x1F83D9AB;
const INIT7: u32 = 0x5BE0CD19;

//for m31 field (2^31-1), split each one to 2 26-bit element
const INIT00_26: u32 = INIT0 & 0x03FFFFFF;
const INIT01_26: u32 = INIT0 >> 26;
const INIT10_26: u32 = INIT1 & 0x03FFFFFF;
const INIT11_26: u32 = INIT1 >> 26;
const INIT20_26: u32 = INIT2 & 0x03FFFFFF;
const INIT21_26: u32 = INIT2 >> 26;
const INIT30_26: u32 = INIT3 & 0x03FFFFFF;
const INIT31_26: u32 = INIT3 >> 26;
const INIT40_26: u32 = INIT4 & 0x03FFFFFF;
const INIT41_26: u32 = INIT4 >> 26;
const INIT50_26: u32 = INIT5 & 0x03FFFFFF;
const INIT51_26: u32 = INIT5 >> 26;
const INIT60_26: u32 = INIT6 & 0x03FFFFFF;
const INIT61_26: u32 = INIT6 >> 26;
const INIT70_26: u32 = INIT7 & 0x03FFFFFF;
const INIT71_26: u32 = INIT7 >> 26;
const _K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
pub struct MyDigest {
    h: [[Variable; 2]; 8],
    nx: usize,
    len: u64,
    kbits: [[Variable; 32]; 64],
    first_8words_h: [[Variable; 2]; 8],
}

impl MyDigest {
    pub fn new<C: Config, B: RootAPI<C>>(api: &mut B) -> Self {
        let mut h = [[api.constant(0); 2]; 8];
        h[0][0] = api.constant(INIT00_26);
        h[0][1] = api.constant(INIT01_26);
        h[1][0] = api.constant(INIT10_26);
        h[1][1] = api.constant(INIT11_26);
        h[2][0] = api.constant(INIT20_26);
        h[2][1] = api.constant(INIT21_26);
        h[3][0] = api.constant(INIT30_26);
        h[3][1] = api.constant(INIT31_26);
        h[4][0] = api.constant(INIT40_26);
        h[4][1] = api.constant(INIT41_26);
        h[5][0] = api.constant(INIT50_26);
        h[5][1] = api.constant(INIT51_26);
        h[6][0] = api.constant(INIT60_26);
        h[6][1] = api.constant(INIT61_26);
        h[7][0] = api.constant(INIT70_26);
        h[7][1] = api.constant(INIT71_26);
        let mut kbits_u8 = [[0; 32]; 64];
        for i in 0..64 {
            for j in 0..32 {
                kbits_u8[i][j] = ((_K[i] >> j) & 1) as u8;
            }
        }
        let mut kbits = [[api.constant(0); 32]; 64];
        for i in 0..64 {
            for j in 0..32 {
                kbits[i][j] = api.constant(kbits_u8[i][j] as u32);
            }
        }
        MyDigest {
            h,
            nx: 0,
            len: 0,
            kbits,
            first_8words_h: h,
        }
    }
    fn reset<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        for i in 0..8 {
            self.h[i] = [api.constant(0); 2];
        }
        self.h[0][0] = api.constant(INIT00_26);
        self.h[0][1] = api.constant(INIT01_26);
        self.h[1][0] = api.constant(INIT10_26);
        self.h[1][1] = api.constant(INIT11_26);
        self.h[2][0] = api.constant(INIT20_26);
        self.h[2][1] = api.constant(INIT21_26);
        self.h[3][0] = api.constant(INIT30_26);
        self.h[3][1] = api.constant(INIT31_26);
        self.h[4][0] = api.constant(INIT40_26);
        self.h[4][1] = api.constant(INIT41_26);
        self.h[5][0] = api.constant(INIT50_26);
        self.h[5][1] = api.constant(INIT51_26);
        self.h[6][0] = api.constant(INIT60_26);
        self.h[6][1] = api.constant(INIT61_26);
        self.h[7][0] = api.constant(INIT70_26);
        self.h[7][1] = api.constant(INIT71_26);
        self.nx = 0;
        self.len = 0;
    }
    //always write a chunk
    fn chunk_write<C: Config, B: RootAPI<C>>(&mut self, api: &mut B, p: &[Variable]) {
        if p.len() != CHUNK || self.nx != 0 {
            panic!("p.len() != CHUNK || self.nx != 0");
        }
        self.len += CHUNK as u64;
        let tmp_h = self.h;
        self.h = self.block(api, tmp_h, p);
    }
    fn chunk_write_compress<C: Config, B: RootAPI<C>>(&mut self, api: &mut B, p: &[Variable]) {
        if p.len() != CHUNK * 8 || self.nx != 0 {
            panic!("p.len() != CHUNK || self.nx != 0");
        }
        self.len += CHUNK as u64;
        let tmp_h = self.h;
        self.h = self.block_37bytes_compress(api, tmp_h, p);
    }
    fn return_sum<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) -> [Variable; SHA256LEN] {
        let mut digest = [api.constant(0); SHA256LEN];

        m31_26_array_put_uint32(api, &mut digest[0..], self.h[0]);
        m31_26_array_put_uint32(api, &mut digest[4..], self.h[1]);
        m31_26_array_put_uint32(api, &mut digest[8..], self.h[2]);
        m31_26_array_put_uint32(api, &mut digest[12..], self.h[3]);
        m31_26_array_put_uint32(api, &mut digest[16..], self.h[4]);
        m31_26_array_put_uint32(api, &mut digest[20..], self.h[5]);
        m31_26_array_put_uint32(api, &mut digest[24..], self.h[6]);
        m31_26_array_put_uint32(api, &mut digest[28..], self.h[7]);
        digest
    }
    fn block<C: Config, B: RootAPI<C>>(
        &mut self,
        api: &mut B,
        h: [[Variable; 2]; 8],
        p: &[Variable],
    ) -> [[Variable; 2]; 8] {
        let mut p = p;
        let mut hh = h;
        while p.len() >= CHUNK {
            let mut msg_schedule = vec![];
            for t in 0..64 {
                if t <= 15 {
                    msg_schedule.push(bytes_to_bits(api, &p[t * 4..t * 4 + 4]));
                } else {
                    let term1_tmp = sigma1(api, &msg_schedule[t - 2]);
                    let term1 = bit_array_to_m31_26(api, &term1_tmp);
                    let term2 = bit_array_to_m31_26(api, &msg_schedule[t - 7]);
                    let term3_tmp = sigma0(api, &msg_schedule[t - 15]);
                    let term3 = bit_array_to_m31_26(api, &term3_tmp);
                    let term4 = bit_array_to_m31_26(api, &msg_schedule[t - 16]);
                    let schedule_tmp1 = big_array_add_no_reduce(api, &term1, &term2);
                    let schedule_tmp2 = big_array_add_no_reduce(api, &term3, &term4);
                    let schedule = big_array_add_no_reduce(api, &schedule_tmp1, &schedule_tmp2);
                    let schedule_bits =
                        m31_26_to_bit_array_seperate(api, &schedule, 2)[..32].to_vec();
                    msg_schedule.push(schedule_bits);
                }
            }
            let mut a = hh[0].to_vec();
            let mut b = hh[1].to_vec();
            let mut c = hh[2].to_vec();
            let mut d = hh[3].to_vec();
            let mut e = hh[4].to_vec();
            let mut f = hh[5].to_vec();
            let mut g = hh[6].to_vec();
            let mut h = hh[7].to_vec();

            //rewrite
            let mut a_bit = m31_26_to_bit_array_seperate(api, &a, 0)[..32].to_vec();
            let mut b_bit = m31_26_to_bit_array_seperate(api, &b, 0)[..32].to_vec();
            let mut c_bit = m31_26_to_bit_array_seperate(api, &c, 0)[..32].to_vec();
            let mut e_bit = m31_26_to_bit_array_seperate(api, &e, 0)[..32].to_vec();
            let mut f_bit = m31_26_to_bit_array_seperate(api, &f, 0)[..32].to_vec();
            let mut g_bit = m31_26_to_bit_array_seperate(api, &g, 0)[..32].to_vec();
            for (t, schedule) in msg_schedule.iter().enumerate().take(64) {
                let mut t1_term1 = [api.constant(0); 2];
                t1_term1[0] = h[0];
                t1_term1[1] = h[1];
                let t1_term2_tmp = cap_sigma1(api, &e_bit);
                let t1_term2 = bit_array_to_m31_26(api, &t1_term2_tmp);
                let t1_term3_tmp = ch(api, &e_bit, &f_bit, &g_bit);
                let t1_term3 = bit_array_to_m31_26(api, &t1_term3_tmp);
                let t1_term4 = bit_array_to_m31_26(api, &self.kbits[t]); //rewrite to [2]frontend.Variable
                let t1_term5 = bit_array_to_m31_26(api, schedule);
                let tmp1 = big_array_add_no_reduce(api, &t1_term1, &t1_term2);
                let tmp2 = big_array_add_no_reduce(api, &t1_term3, &t1_term4);
                let tmp3 = big_array_add_no_reduce(api, &tmp1, &tmp2);
                let tmp4 = big_array_add_no_reduce(api, &tmp3, &t1_term5);
                let t1 = tmp4;
                let t2_tmp1 = cap_sigma0(api, &a_bit);
                let t2_tmp2 = bit_array_to_m31_26(api, &t2_tmp1);
                let t2_tmp3 = maj(api, &a_bit, &b_bit, &c_bit);
                let t2_tmp4 = bit_array_to_m31_26(api, &t2_tmp3);
                let t2 = big_array_add_no_reduce(api, &t2_tmp2, &t2_tmp4);
                let new_a_bit_tmp = big_array_add_no_reduce(api, &t1, &t2);
                let new_a_bit = m31_26_to_bit_array_seperate(api, &new_a_bit_tmp, 4)[..32].to_vec();
                let new_e_bit_tmp = big_array_add_no_reduce(api, &d, &t1);
                let new_e_bit = m31_26_to_bit_array_seperate(api, &new_e_bit_tmp, 4)[..32].to_vec();
                h = g;
                g = f;
                f = e;
                d = c;
                c = b;
                b = a;
                a = bit_array_to_m31_26(api, &new_a_bit).to_vec();
                // a = new_a_bit_tmp;
                e = bit_array_to_m31_26(api, &new_e_bit).to_vec();
                // e = new_e_bit_tmp;
                g_bit = f_bit;
                f_bit = e_bit;
                c_bit = b_bit;
                b_bit = a_bit;
                a_bit = new_a_bit;
                e_bit = new_e_bit;
            }
            hh[0] = sha_m31_26_add(api, &hh[0], &a, 26);
            hh[1] = sha_m31_26_add(api, &hh[1], &b, 26);
            hh[2] = sha_m31_26_add(api, &hh[2], &c, 26);
            hh[3] = sha_m31_26_add(api, &hh[3], &d, 26);
            hh[4] = sha_m31_26_add(api, &hh[4], &e, 26);
            hh[5] = sha_m31_26_add(api, &hh[5], &f, 26);
            hh[6] = sha_m31_26_add(api, &hh[6], &g, 26);
            hh[7] = sha_m31_26_add(api, &hh[7], &h, 26);
            p = &p[CHUNK..];
        }
        hh
    }
    //consider in a 64-byte block, only 8-th word is different
    //so we can skip the 0~7-th word when doing second part
    fn block_37bytes_compress<C: Config, B: RootAPI<C>>(
        &mut self,
        api: &mut B,
        h: [[Variable; 2]; 8],
        p: &[Variable],
    ) -> [[Variable; 2]; 8] {
        let mut hh = h;
        let mut msg_schedule = vec![];
        for t in 0..16 {
            msg_schedule.push(p[t * 32..t * 32 + 32].to_vec());
        }
        for t in 16..64 {
            let term1_tmp = sigma1(api, &msg_schedule[t - 2]);
            let term1 = bit_array_to_m31_26(api, &term1_tmp);
            let term2 = bit_array_to_m31_26(api, &msg_schedule[t - 7]);
            let term3_tmp = sigma0(api, &msg_schedule[t - 15]);
            let term3 = bit_array_to_m31_26(api, &term3_tmp);
            let term4 = bit_array_to_m31_26(api, &msg_schedule[t - 16]);
            let schedule_tmp1 = big_array_add_no_reduce(api, &term1, &term2);
            let schedule_tmp2 = big_array_add_no_reduce(api, &term3, &term4);
            let schedule = big_array_add_no_reduce(api, &schedule_tmp1, &schedule_tmp2);
            let schedule_bits = m31_26_to_bit_array_seperate(api, &schedule, 2)[..32].to_vec();
            msg_schedule.push(schedule_bits);
        }
        let mut a = self.first_8words_h[0].to_vec();
        let mut b = self.first_8words_h[1].to_vec();
        let mut c = self.first_8words_h[2].to_vec();
        let mut d = self.first_8words_h[3].to_vec();
        let mut e = self.first_8words_h[4].to_vec();
        let mut f = self.first_8words_h[5].to_vec();
        let mut g = self.first_8words_h[6].to_vec();
        let mut h = self.first_8words_h[7].to_vec();

        //rewrite
        let mut a_bit = m31_26_to_bit_array_seperate(api, &a, 0)[..32].to_vec();
        let mut b_bit = m31_26_to_bit_array_seperate(api, &b, 0)[..32].to_vec();
        let mut c_bit = m31_26_to_bit_array_seperate(api, &c, 0)[..32].to_vec();
        let mut e_bit = m31_26_to_bit_array_seperate(api, &e, 0)[..32].to_vec();
        let mut f_bit = m31_26_to_bit_array_seperate(api, &f, 0)[..32].to_vec();
        let mut g_bit = m31_26_to_bit_array_seperate(api, &g, 0)[..32].to_vec();
        for (t, schedule) in msg_schedule.iter().enumerate().skip(8).take(64) {
            let mut t1_term1 = [api.constant(0); 2];
            t1_term1[0] = h[0];
            t1_term1[1] = h[1];
            let t1_term2_tmp = cap_sigma1(api, &e_bit);
            let t1_term2 = bit_array_to_m31_26(api, &t1_term2_tmp);
            let t1_term3_tmp = ch(api, &e_bit, &f_bit, &g_bit);
            let t1_term3 = bit_array_to_m31_26(api, &t1_term3_tmp);
            let t1_term4 = bit_array_to_m31_26(api, &self.kbits[t]); //rewrite to [2]frontend.Variable
            let t1_term5 = bit_array_to_m31_26(api, schedule);
            let tmp1 = big_array_add_no_reduce(api, &t1_term1, &t1_term2);
            let tmp2 = big_array_add_no_reduce(api, &t1_term3, &t1_term4);
            let tmp3 = big_array_add_no_reduce(api, &tmp1, &tmp2);
            let tmp4 = big_array_add_no_reduce(api, &tmp3, &t1_term5);
            let t1 = tmp4;
            let t2_tmp1 = cap_sigma0(api, &a_bit);
            let t2_tmp2 = bit_array_to_m31_26(api, &t2_tmp1);
            let t2_tmp3 = maj(api, &a_bit, &b_bit, &c_bit);
            let t2_tmp4 = bit_array_to_m31_26(api, &t2_tmp3);
            let t2 = big_array_add_no_reduce(api, &t2_tmp2, &t2_tmp4);
            let new_a_bit_tmp = big_array_add_no_reduce(api, &t1, &t2);
            let new_a_bit = m31_26_to_bit_array_seperate(api, &new_a_bit_tmp, 4)[..32].to_vec();
            let new_e_bit_tmp = big_array_add_no_reduce(api, &d, &t1);
            let new_e_bit = m31_26_to_bit_array_seperate(api, &new_e_bit_tmp, 4)[..32].to_vec();
            h = g;
            g = f;
            f = e;
            d = c;
            c = b;
            b = a;
            a = bit_array_to_m31_26(api, &new_a_bit).to_vec();
            e = bit_array_to_m31_26(api, &new_e_bit).to_vec();
            g_bit = f_bit;
            f_bit = e_bit;
            c_bit = b_bit;
            b_bit = a_bit;
            a_bit = new_a_bit;
            e_bit = new_e_bit;
        }
        hh[0] = sha_m31_26_add(api, &hh[0], &a, 26);
        hh[1] = sha_m31_26_add(api, &hh[1], &b, 26);
        hh[2] = sha_m31_26_add(api, &hh[2], &c, 26);
        hh[3] = sha_m31_26_add(api, &hh[3], &d, 26);
        hh[4] = sha_m31_26_add(api, &hh[4], &e, 26);
        hh[5] = sha_m31_26_add(api, &hh[5], &f, 26);
        hh[6] = sha_m31_26_add(api, &hh[6], &g, 26);
        hh[7] = sha_m31_26_add(api, &hh[7], &h, 26);
        hh
    }
    fn block_37bytes_compress_26_set_8words_states<C: Config, B: RootAPI<C>>(
        &mut self,
        api: &mut B,
        h: [[Variable; 2]; 8],
        p: &[Variable],
    ) {
        let hh = h;
        let mut msg_schedule = vec![];
        for t in 0..8 {
            msg_schedule.push(p[t * 32..t * 32 + 32].to_vec());
        }
        let mut a = hh[0].to_vec();
        let mut b = hh[1].to_vec();
        let mut c = hh[2].to_vec();
        let mut d = hh[3].to_vec();
        let mut e = hh[4].to_vec();
        let mut f = hh[5].to_vec();
        let mut g = hh[6].to_vec();
        let mut h = hh[7].to_vec();

        //rewrite
        let mut a_bit = m31_26_to_bit_array_seperate(api, &a, 0)[..32].to_vec();
        let mut b_bit = m31_26_to_bit_array_seperate(api, &b, 0)[..32].to_vec();
        let mut c_bit = m31_26_to_bit_array_seperate(api, &c, 0)[..32].to_vec();
        let mut e_bit = m31_26_to_bit_array_seperate(api, &e, 0)[..32].to_vec();
        let mut f_bit = m31_26_to_bit_array_seperate(api, &f, 0)[..32].to_vec();
        let mut g_bit = m31_26_to_bit_array_seperate(api, &g, 0)[..32].to_vec();
        for (t, schedule) in msg_schedule.iter().enumerate().take(8) {
            let mut t1_term1 = [api.constant(0); 2];
            t1_term1[0] = h[0];
            t1_term1[1] = h[1];
            let t1_term2_tmp = cap_sigma1(api, &e_bit);
            let t1_term2 = bit_array_to_m31_26(api, &t1_term2_tmp);
            let t1_term3_tmp = ch(api, &e_bit, &f_bit, &g_bit);
            let t1_term3 = bit_array_to_m31_26(api, &t1_term3_tmp);
            let t1_term4 = bit_array_to_m31_26(api, &self.kbits[t]); //rewrite to [2]frontend.Variable
            let t1_term5 = bit_array_to_m31_26(api, schedule);
            let tmp1 = big_array_add_no_reduce(api, &t1_term1, &t1_term2);
            let tmp2 = big_array_add_no_reduce(api, &t1_term3, &t1_term4);
            let tmp3 = big_array_add_no_reduce(api, &tmp1, &tmp2);
            let tmp4 = big_array_add_no_reduce(api, &tmp3, &t1_term5);
            let t1 = tmp4;
            let t2_tmp1 = cap_sigma0(api, &a_bit);
            let t2_tmp2 = bit_array_to_m31_26(api, &t2_tmp1);
            let t2_tmp3 = maj(api, &a_bit, &b_bit, &c_bit);
            let t2_tmp4 = bit_array_to_m31_26(api, &t2_tmp3);
            let t2 = big_array_add_no_reduce(api, &t2_tmp2, &t2_tmp4);
            let new_a_bit_tmp = big_array_add_no_reduce(api, &t1, &t2);
            let new_a_bit = m31_26_to_bit_array_seperate(api, &new_a_bit_tmp, 4)[..32].to_vec();
            let new_e_bit_tmp = big_array_add_no_reduce(api, &d, &t1);
            let new_e_bit = m31_26_to_bit_array_seperate(api, &new_e_bit_tmp, 4)[..32].to_vec();
            h = g;
            g = f;
            f = e;
            d = c;
            c = b;
            b = a;
            a = bit_array_to_m31_26(api, &new_a_bit).to_vec();
            e = bit_array_to_m31_26(api, &new_e_bit).to_vec();
            g_bit = f_bit;
            f_bit = e_bit;
            c_bit = b_bit;
            b_bit = a_bit;
            a_bit = new_a_bit;
            e_bit = new_e_bit;
        }
        self.first_8words_h[0] = a.try_into().unwrap();
        self.first_8words_h[1] = b.try_into().unwrap();
        self.first_8words_h[2] = c.try_into().unwrap();
        self.first_8words_h[3] = d.try_into().unwrap();
        self.first_8words_h[4] = e.try_into().unwrap();
        self.first_8words_h[5] = f.try_into().unwrap();
        self.first_8words_h[6] = g.try_into().unwrap();
        self.first_8words_h[7] = h.try_into().unwrap();
    }
}

pub fn sha256_37bytes<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    orign_data: &[Variable],
) -> Vec<Variable> {
    let mut data = orign_data.to_vec();
    let n = data.len();
    if n != 32 + 1 + 4 {
        panic!("len(orignData) !=  32+1+4")
    }
    let mut pre_pad = vec![builder.constant(0); 64 - 37];
    pre_pad[0] = builder.constant(128); //0x80
    pre_pad[64 - 37 - 2] = builder.constant((37) * 8 / 256); //length byte
    pre_pad[64 - 37 - 1] = builder.constant((32 + 1 + 4) * 8 - 256); //length byte
    data.append(&mut pre_pad); //append padding

    let mut d = MyDigest::new(builder);
    d.chunk_write(builder, &data);
    d.return_sum(builder).to_vec()
}

pub fn sha256_37bytes_compress<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    orign_data: &[Variable],
) -> Vec<Variable> {
    let mut data = orign_data.to_vec();
    let n = data.len();
    if n != (32 + 1 + 4) * 8 {
        panic!("len(orignData) !=  37 bytes")
    }
    let mut pre_pad = vec![builder.constant(0); (64 - 37) * 8];
    pre_pad[0] = builder.constant(1); //0x80
    pre_pad[207] = builder.constant(1); //length byte
    pre_pad[210] = builder.constant(1); //length byte
    pre_pad[212] = builder.constant(1); //length byte
    data.append(&mut pre_pad); //append padding
    data[36 * 8..40 * 8].reverse();
    data[60 * 8..64 * 8].reverse();
    let mut d = MyDigest::new(builder);
    d.block_37bytes_compress_26_set_8words_states(builder, d.h, &data);
    d.chunk_write_compress(builder, &data);
    d.return_sum(builder).to_vec()
}

pub fn sha256_37bytes_256batch_compress<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    inputs: &[Vec<Variable>],
) -> Vec<Vec<Variable>> {
    let mut d = MyDigest::new(builder);
    let mut outputs = vec![];
    for (i, input) in inputs.iter().enumerate() {
        let mut pre_pad = vec![builder.constant(0); (64 - 37) * 8];
        pre_pad[0] = builder.constant(1); //0x80
        pre_pad[207] = builder.constant(1); //length byte
        pre_pad[210] = builder.constant(1); //length byte
        pre_pad[212] = builder.constant(1); //length byte
        let mut data = input.clone();
        let n = data.len();
        if n != (32 + 1 + 4) * 8 {
            panic!("len(orignData) !=  37 bytes")
        }
        data.append(&mut pre_pad); //append padding
        data[36 * 8..40 * 8].reverse();
        data[60 * 8..64 * 8].reverse();
        if i == 0 {
            d.block_37bytes_compress_26_set_8words_states(builder, d.h, &data);
        }
        d.chunk_write_compress(builder, &data);
        outputs.push(d.return_sum(builder).to_vec());
        d.reset(builder);
    }
    outputs
}

pub fn sha256_var_bytes<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    orign_data: &[Variable],
) -> Vec<Variable> {
    let mut data = orign_data.to_vec();
    let n = data.len();
    let n_bytes = (n * 8).to_be_bytes().to_vec();
    let mut pad;
    if n % 64 > 55 {
        //need to add one more chunk (64bytes)
        pad = vec![builder.constant(0); 128 - n % 64];
        pad[0] = builder.constant(128); //0x80
    } else {
        pad = vec![builder.constant(0); 64 - n % 64];
        pad[0] = builder.constant(128); //0x80
    }
    let pad_len = pad.len();
    for i in 0..n_bytes.len() {
        pad[pad_len - n_bytes.len() + i] = builder.constant(n_bytes[i] as u32);
    }
    data.append(&mut pad); //append padding

    let mut d = MyDigest::new(builder);
    d.reset(builder);

    let n = data.len();
    for i in 0..n / 64 {
        d.chunk_write(builder, &data[i * 64..(i + 1) * 64]);
    }
    d.return_sum(builder).to_vec()
}

pub fn check_sha256_37bytes<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    origin_data: &[Variable],
) -> Vec<Variable> {
    let output = origin_data[37..].to_vec();
    let result = sha256_37bytes(builder, &origin_data[..37]);
    for i in 0..32 {
        builder.assert_is_equal(result[i], output[i]);
    }
    result
}

pub fn check_sha256_37bytes_compress<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    origin_data: &[Variable],
) -> Vec<Variable> {
    let output = origin_data[37 * 8..].to_vec();
    let result = sha256_37bytes_compress(builder, &origin_data[..37 * 8]);
    for i in 0..32 {
        builder.assert_is_equal(result[i], output[i]);
    }
    result
}

pub fn check_sha256_37bytes_256batch_compress<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    inputs: &[Vec<Variable>],
    outputs: &[Vec<Variable>],
) -> Vec<Vec<Variable>> {
    let result = sha256_37bytes_256batch_compress(builder, inputs);
    for i in 0..outputs.len() {
        for j in 0..32 {
            builder.assert_is_equal(result[i][j], outputs[i][j]);
        }
    }
    result
}
