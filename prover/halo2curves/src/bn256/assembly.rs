macro_rules! field_arithmetic_asm {
    (
        $field:ident,
        $modulus:ident,
        $inv:ident
    ) => {
        use std::arch::asm;

        impl $field {
            /// Doubles this field element.
            #[inline]
            pub fn double(&self) -> $field {
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    asm!(
                        // load a array to former registers
                        "mov r8, qword ptr [{a_ptr} + 0]",
                        "mov r9, qword ptr [{a_ptr} + 8]",
                        "mov r10, qword ptr [{a_ptr} + 16]",
                        "mov r11, qword ptr [{a_ptr} + 24]",

                        // // add a array and b array with carry
                        "add r8, r8",
                        "adc r9, r9",
                        "adc r10, r10",
                        "adc r11, r11",

                        // copy result array to latter registers
                        "mov r12, r8",
                        "mov r13, r9",
                        "mov r14, r10",
                        "mov r15, r11",

                        // mod reduction
                        "sub r12, qword ptr [{m_ptr} + 0]",
                        "sbb r13, qword ptr [{m_ptr} + 8]",
                        "sbb r14, qword ptr [{m_ptr} + 16]",
                        "sbb r15, qword ptr [{m_ptr} + 24]",

                        // if carry copy former registers to out areas
                        "cmovc r12, r8",
                        "cmovc r13, r9",
                        "cmovc r14, r10",
                        "cmovc r15, r11",

                        m_ptr = in(reg) $modulus.0.as_ptr(),
                        a_ptr = in(reg) self.0.as_ptr(),
                        out("r8") _,
                        out("r9") _,
                        out("r10") _,
                        out("r11") _,
                        out("r12") r0,
                        out("r13") r1,
                        out("r14") r2,
                        out("r15") r3,
                        options(pure, readonly, nostack)
                    );
                }
                $field([r0, r1, r2, r3])
            }

            /// Squares this element.
            #[inline]
            pub fn square(&self) -> $field {
                self.mul(self)
            }

            #[inline(always)]
            pub(crate) fn montgomery_reduce_256(&self) -> $field {
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;

                unsafe {
                    asm!(
                        "mov r8, qword ptr [{a_ptr} + 0]",
                        "mov r9, qword ptr [{a_ptr} + 8]",
                        "mov r10, qword ptr [{a_ptr} + 16]",
                        "mov r11, qword ptr [{a_ptr} + 24]",
                        "mov r15, {inv}",
                        "xor r12, r12",

                        // i0
                        "mov rdx, r8",
                        "mulx rcx, rdx, r15",

                        // j0
                        "mulx rcx, rax, qword ptr [{m_ptr} + 0]",
                        "adox r8, rax",
                        "adcx r9, rcx",
                        // j1
                        "mulx rcx, rax, qword ptr [{m_ptr} + 8]",
                        "adox r9, rax",
                        "adcx r10, rcx",
                        // j2
                        "mulx rcx, rax, qword ptr [{m_ptr} + 16]",
                        "adox r10, rax",
                        "adcx r11, rcx",
                        // j3
                        "mulx rcx, rax, qword ptr [{m_ptr} + 24]",
                        "adox r11, rax",
                        "adcx r8, rcx",
                        "adox r8, r12",

                        // i1
                        "mov rdx, r9",
                        "mulx rcx, rdx, r15",

                        // j0
                        "mulx rcx, rax, qword ptr [{m_ptr} + 0]",
                        "adox r9, rax",
                        "adcx r10, rcx",

                        // j1
                        "mulx rcx, rax, qword ptr [{m_ptr} + 8]",
                        "adox r10, rax",
                        "adcx r11, rcx",
                        // j2
                        "mulx rcx, rax, qword ptr [{m_ptr} + 16]",
                        "adox r11, rax",
                        "adcx r8, rcx",
                        // j3
                        "mulx rcx, rax, qword ptr [{m_ptr} + 24]",
                        "adox r8, rax",
                        "adcx r9, rcx",
                        "adox r9, r12",

                        // i2
                        "mov rdx, r10",
                        "mulx rcx, rdx, r15",

                        // j0
                        "mulx rcx, rax, qword ptr [{m_ptr} + 0]",
                        "adox r10, rax",
                        "adcx r11, rcx",

                        // j1
                        "mulx rcx, rax, qword ptr [{m_ptr} + 8]",
                        "adox r11, rax",
                        "adcx r8, rcx",

                        // j2
                        "mulx rcx, rax, qword ptr [{m_ptr} + 16]",
                        "adox r8, rax",
                        "adcx r9, rcx",

                        // j3
                        "mulx rcx, rax, qword ptr [{m_ptr} + 24]",
                        "adox r9, rax",
                        "adcx r10, rcx",
                        "adox r10, r12",

                        // i3
                        "mov rdx, r11",
                        "mulx rcx, rdx, r15",
                        // j0
                        "mulx rcx, rax, qword ptr [{m_ptr} + 0]",
                        "adox r11, rax",
                        "adcx r8, rcx",
                        // j1
                        "mulx rcx, rax, qword ptr [{m_ptr} + 8]",
                        "adox r8, rax",
                        "adcx r9, rcx",
                        // j2
                        "mulx rcx, rax, qword ptr [{m_ptr} + 16]",
                        "adox r9, rax",
                        "adcx r10, rcx",
                        // j3
                        "mulx rcx, rax, qword ptr [{m_ptr} + 24]",
                        "adox r10, rax",
                        "adcx r11, rcx",
                        "adox r11, r12",

                        // modular reduction is not required since:
                        // high(inv * p3) + 2 < p3

                        a_ptr = in(reg) self.0.as_ptr(),
                        m_ptr = in(reg) $modulus.0.as_ptr(),
                        inv = in(reg) $inv,

                        out("rax") _,
                        out("rcx") _,
                        out("rdx") _,
                        out("r8") r0,
                        out("r9") r1,
                        out("r10") r2,
                        out("r11") r3,
                        out("r12") _,
                        out("r13") _,
                        out("r14") _,
                        out("r15") _,
                        options(pure, readonly, nostack)
                    )
                }
                $field([r0, r1, r2, r3])
            }

            /// Multiplies `rhs` by `self`, returning the result.
            #[inline]
            pub fn mul(&self, rhs: &Self) -> $field {
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    asm!(

                        // Coarsely Integrated Operand Scanning:
                        // - Analyzing and Comparing Montgomery Multiplication Algorithms
                        //   Cetin Kaya Koc and Tolga Acar and Burton S. Kaliski Jr.
                        //   http://pdfs.semanticscholar.org/5e39/41ff482ec3ee41dc53c3298f0be085c69483.pdf
                        //
                        // No-carry optimization
                        // - https://hackmd.io/@gnark/modular_multiplication
                        //
                        // Code generator
                        // - https://github.com/mratsim/constantine/blob/151f284/constantine/math/arithmetic/assembly/limbs_asm_mul_mont_x86_adx_bmi2.nim#L231-L269
                        //
                        // Assembly generated
                        // - https://github.com/ethereum/evmone/blob/d006d81/lib/evmmax/mulMont256_spare_bits_asm_adx.S

                        // Algorithm
                        // -----------------------------------------
                        // for i=0 to N-1
                        //   for j=0 to N-1
                        // 		(A,t[j])  := t[j] + a[j]*b[i] + A
                        //   m := t[0]*m0ninv mod W
                        // 	C,_ := t[0] + m*M[0]
                        // 	for j=1 to N-1
                        // 		(C,t[j-1]) := t[j] + m*M[j] + C
                        //   t[N-1] = C + A

                        // Outer loop i = 0
                        //   Multiplication
                        "mov  rdx, qword ptr [{b_ptr} + 0]",
                        "mulx r13, r11, qword ptr [{a_ptr} + 0]",
                        "mulx rax, r15, qword ptr [{a_ptr} + 8]",
                        "add  r13, r15",
                        "mulx r14, r15, qword ptr [{a_ptr} + 16]",
                        "adc  rax, r15",
                        //   Multiplication. last limb
                        "mulx r12, r15, qword ptr [{a_ptr} + 24]",
                        "adc  r14, r15",
                        "adc  r12, 0", // accumulate last carries in hi word

                        //   Reduction
                        //   m = t[0] * m0ninv mod 2^w
                        "mov  rdx, r11",
                        "imul rdx, {inv}",
                        "xor  r15, r15",
                        //   C,_ := t[0] + m*M[0]
                        "mulx r15, r10, qword ptr [{m_ptr} + 0]",
                        "adcx r10, r11",
                        "mov  r11, r15",
                        "mov  r10, 0",
                        //   for j=1 to N-1
                        //     (C, t[j-1]) := t[j] + m*M[j] + C
                        "adcx r11, r13",
                        "mulx r13, r15, qword ptr [{m_ptr} + 8]",
                        "adox r11, r15",
                        "adcx r13, rax",
                        "mulx rax, r15, qword ptr [{m_ptr} + 16]",
                        "adox r13, r15",
                        "adcx rax, r14",
                        "mulx r14, r15, qword ptr [{m_ptr} + 24]",
                        "adox rax, r15",
                        //   Reduction carry
                        "adcx r10, r12",
                        "adox r14, r10",

                        // Outer loop i = 1, j in [0, 4)
                        "mov  rdx, qword ptr [{b_ptr} + 8]",
                        "xor  r12, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 0]",
                        "adox r11, r15",
                        "adcx r13, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 8]",
                        "adox r13, r15",
                        "adcx rax, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 16]",
                        "adox rax, r15",
                        "adcx r14, r12",
                        //   Multiplication, last limb
                        "mulx r12, r15, qword ptr [{a_ptr} + 24]",
                        "adox r14, r15",
                        "mov  rdx, 0", // accumulate last carries in hi word
                        "adcx r12, rdx",
                        "adox r12, rdx",

                        //   Reduction
                        //   m = t[0] * m0ninv mod 2^w
                        "mov  rdx, r11",
                        "imul rdx, {inv}",
                        "xor  r15, r15",
                        //   C,_ := t[0] + m*M[0]
                        "mulx r15, r10, qword ptr [{m_ptr} + 0]",
                        "adcx r10, r11",
                        "mov  r11, r15",
                        "mov  r10, 0",
                        //   for j=1 to N-1
                        //     (C, t[j-1]) := t[j] + m*M[j] + C
                        "adcx r11, r13",
                        "mulx r13, r15, qword ptr [{m_ptr} + 8]",
                        "adox r11, r15",
                        "adcx r13, rax",
                        "mulx rax, r15, qword ptr [{m_ptr} + 16]",
                        "adox r13, r15",
                        "adcx rax, r14",
                        "mulx r14, r15, qword ptr [{m_ptr} + 24]",
                        "adox rax, r15",
                        //   Reduction carry
                        "adcx r10, r12",
                        "adox r14, r10",

                        // Outer loop i = 2, j in [0, 4)
                        "mov  rdx, qword ptr [{b_ptr} + 16]",
                        "xor  r12, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 0]",
                        "adox r11, r15",
                        "adcx r13, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 8]",
                        "adox r13, r15",
                        "adcx rax, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 16]",
                        "adox rax, r15",
                        "adcx r14, r12",
                        //   Multiplication, last limb
                        "mulx r12, r15, qword ptr [{a_ptr} + 24]",
                        "adox r14, r15",
                        "mov  rdx, 0", // accumulate last carries in hi word
                        "adcx r12, rdx",
                        "adox r12, rdx",

                        //   Reduction
                        //   m = t[0] * m0ninv mod 2^w
                        "mov  rdx, r11",
                        "imul rdx, {inv}",
                        "xor  r15, r15",
                        //   C,_ := t[0] + m*M[0]
                        "mulx r15, r10, qword ptr [{m_ptr} + 0]",
                        "adcx r10, r11",
                        "mov  r11, r15",
                        "mov  r10, 0",
                        //   for j=1 to N-1
                        //     (C, t[j-1]) := t[j] + m*M[j] + C
                        "adcx r11, r13",
                        "mulx r13, r15, qword ptr [{m_ptr} + 8]",
                        "adox r11, r15",
                        "adcx r13, rax",
                        "mulx rax, r15, qword ptr [{m_ptr} + 16]",
                        "adox r13, r15",
                        "adcx rax, r14",
                        "mulx r14, r15, qword ptr [{m_ptr} + 24]",
                        "adox rax, r15",
                        //   Reduction carry
                        "adcx r10, r12",
                        "adox r14, r10",

                        // Outer loop i = 3, j in [0, 4)
                        "mov  rdx, qword ptr [{b_ptr} + 24]",
                        "xor  r12, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 0]",
                        "adox r11, r15",
                        "adcx r13, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 8]",
                        "adox r13, r15",
                        "adcx rax, r12",
                        "mulx r12, r15, qword ptr [{a_ptr} + 16]",
                        "adox rax, r15",
                        "adcx r14, r12",
                        //   Multiplication, last limb
                        "mulx r12, r15, qword ptr [{a_ptr} + 24]",
                        "adox r14, r15",
                        "mov  rdx, 0", // accumulate last carries in hi word
                        "adcx r12, rdx",
                        "adox r12, rdx",

                        //   Reduction
                        //   m = t[0] * m0ninv mod 2^w
                        "mov  rdx, r11",
                        "imul rdx, {inv}",
                        "xor  r15, r15",
                        //   C,_ := t[0] + m*M[0]
                        "mulx r15, r10, qword ptr [{m_ptr} + 0]",
                        "adcx r10, r11",
                        "mov  r11, r15",
                        "mov  r10, 0",
                        //   for j=1 to N-1
                        //     (C, t[j-1]) := t[j] + m*M[j] + C
                        "adcx r11, r13",
                        "mulx r13, r15, qword ptr [{m_ptr} + 8]",
                        "adox r11, r15",
                        "adcx r13, rax",
                        "mulx rax, r15, qword ptr [{m_ptr} + 16]",
                        "adox r13, r15",
                        "adcx rax, r14",
                        "mulx r14, r15, qword ptr [{m_ptr} + 24]",
                        "adox rax, r15",
                        //   Reduction carry
                        "adcx r10, r12",
                        "adox r14, r10",

                        //   Final subtraction
                        "mov  r12, r11",
                        "sub  r12, qword ptr [{m_ptr} + 0]",
                        "mov  r10, r13",
                        "sbb  r10, qword ptr [{m_ptr} + 8]",
                        "mov  rdx, rax",
                        "sbb  rdx, qword ptr [{m_ptr} + 16]",
                        "mov  r15, r14",
                        "sbb  r15, qword ptr [{m_ptr} + 24]",

                        "cmovnc r11, r12",
                        "cmovnc r13, r10",
                        "cmovnc rax, rdx",
                        "cmovnc r14, r15",

                        m_ptr = in(reg) $modulus.0.as_ptr(),
                        a_ptr = in(reg) self.0.as_ptr(),
                        b_ptr = in(reg) rhs.0.as_ptr(),
                        inv = in(reg) $inv,
                        out("rax") r2,
                        out("rdx") _,
                        out("r10") _,
                        out("r11") r0,
                        out("r12") _,
                        out("r13") r1,
                        out("r14") r3,
                        out("r15") _,
                        options(pure, readonly, nostack)
                    )
                }

                $field([r0, r1, r2, r3])
            }

            /// Subtracts `rhs` from `self`, returning the result.
            #[inline]
            pub fn sub(&self, rhs: &Self) -> $field {
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    asm!(
                        // init modulus area
                        "mov r12, qword ptr [{m_ptr} + 0]",
                        "mov r13, qword ptr [{m_ptr} + 8]",
                        "mov r14, qword ptr [{m_ptr} + 16]",
                        "mov r15, qword ptr [{m_ptr} + 24]",

                        // load a array to former registers
                        "mov r8, qword ptr [{a_ptr} + 0]",
                        "mov r9, qword ptr [{a_ptr} + 8]",
                        "mov r10, qword ptr [{a_ptr} + 16]",
                        "mov r11, qword ptr [{a_ptr} + 24]",

                        // sub a array and b array with borrow
                        "sub r8, qword ptr [{b_ptr} + 0]",
                        "sbb r9, qword ptr [{b_ptr} + 8]",
                        "sbb r10, qword ptr [{b_ptr} + 16]",
                        "sbb r11, qword ptr [{b_ptr} + 24]",

                        // Mask: rax contains 0xFFFF if < m or 0x0000 otherwise
                        "sbb rax, rax",

                        // Zero-out the modulus if a-b < m or leave as-is otherwise
                        "and r12, rax",
                        "and r13, rax",
                        "and r14, rax",
                        "and r15, rax",

                        // Add zero if a-b < m or a-b+m otherwise
                        "add  r12, r8",
                        "adc  r13, r9",
                        "adc  r14, r10",
                        "adc  r15, r11",

                        m_ptr = in(reg) $modulus.0.as_ptr(),
                        a_ptr = in(reg) self.0.as_ptr(),
                        b_ptr = in(reg) rhs.0.as_ptr(),
                        out("rax") _,
                        out("r8") _,
                        out("r9") _,
                        out("r10") _,
                        out("r11") _,
                        out("r12") r0,
                        out("r13") r1,
                        out("r14") r2,
                        out("r15") r3,
                        options(pure, readonly, nostack)
                    );
                }
                $field([r0, r1, r2, r3])
            }

            /// Adds `rhs` to `self`, returning the result.
            #[inline]
            pub fn add(&self, rhs: &Self) -> $field {
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    asm!(
                        // load a array to former registers
                        "mov r8, qword ptr [{a_ptr} + 0]",
                        "mov r9, qword ptr [{a_ptr} + 8]",
                        "mov r10, qword ptr [{a_ptr} + 16]",
                        "mov r11, qword ptr [{a_ptr} + 24]",

                        // add a array and b array with carry
                        "add r8, qword ptr [{b_ptr} + 0]",
                        "adc r9, qword ptr [{b_ptr} + 8]",
                        "adc r10, qword ptr [{b_ptr} + 16]",
                        "adc r11, qword ptr [{b_ptr} + 24]",

                        // copy result array to latter registers
                        "mov r12, r8",
                        "mov r13, r9",
                        "mov r14, r10",
                        "mov r15, r11",

                        // mod reduction
                        "sub r12, qword ptr [{m_ptr} + 0]",
                        "sbb r13, qword ptr [{m_ptr} + 8]",
                        "sbb r14, qword ptr [{m_ptr} + 16]",
                        "sbb r15, qword ptr [{m_ptr} + 24]",

                        // if carry copy former registers to out areas
                        "cmovc r12, r8",
                        "cmovc r13, r9",
                        "cmovc r14, r10",
                        "cmovc r15, r11",

                        m_ptr = in(reg) $modulus.0.as_ptr(),
                        a_ptr = in(reg) self.0.as_ptr(),
                        b_ptr = in(reg) rhs.0.as_ptr(),
                        out("r8") _,
                        out("r9") _,
                        out("r10") _,
                        out("r11") _,
                        out("r12") r0,
                        out("r13") r1,
                        out("r14") r2,
                        out("r15") r3,
                        options(pure, readonly, nostack)
                    );
                }
                $field([r0, r1, r2, r3])
            }

            /// Negates `self`.
            #[inline]
            pub fn neg(&self) -> $field {
                let mut r0: u64;
                let mut r1: u64;
                let mut r2: u64;
                let mut r3: u64;
                unsafe {
                    asm!(
                        // load a array to former registers
                        "mov r8, qword ptr [{m_ptr} + 0]",
                        "mov r9, qword ptr [{m_ptr} + 8]",
                        "mov r10, qword ptr [{m_ptr} + 16]",
                        "mov r11, qword ptr [{m_ptr} + 24]",

                        "sub r8, qword ptr [{a_ptr} + 0]",
                        "sbb r9, qword ptr [{a_ptr} + 8]",
                        "sbb r10, qword ptr [{a_ptr} + 16]",
                        "sbb r11, qword ptr [{a_ptr} + 24]",

                        "mov r12, qword ptr [{a_ptr} + 0]",
                        "mov r13, qword ptr [{a_ptr} + 8]",
                        "mov r14, qword ptr [{a_ptr} + 16]",
                        "mov r15, qword ptr [{a_ptr} + 24]",

                        "or r12, r13",
                        "or r14, r15",
                        "or r12, r14",

                        "mov r13, 0xffffffffffffffff",
                        "cmp r12, 0x0000000000000000",
                        "cmove r13, r12",

                        "and r8, r13",
                        "and r9, r13",
                        "and r10, r13",
                        "and r11, r13",

                        a_ptr = in(reg) self.0.as_ptr(),
                        m_ptr = in(reg) $modulus.0.as_ptr(),
                        out("r8") r0,
                        out("r9") r1,
                        out("r10") r2,
                        out("r11") r3,
                        out("r12") _,
                        out("r13") _,
                        out("r14") _,
                        out("r15") _,
                        options(pure, readonly, nostack)
                    )
                }
                $field([r0, r1, r2, r3])
            }
        }

        impl From<$field> for [u64; 4] {
            fn from(elt: $field) -> [u64; 4] {
                // Turn into canonical form by computing
                // (a.R) / R = a
                elt.montgomery_reduce_256().0
            }
        }
    };
}

pub(crate) use field_arithmetic_asm;
