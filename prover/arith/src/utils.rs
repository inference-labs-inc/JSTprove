// NOTE(HS) acknowledge to https://github.com/EugeneGonzalez/bit_reverse
#[rustfmt::skip]
const U8_REVERSE_LOOKUP: [u8; 256] = [
    0,  128, 64, 192, 32, 160,  96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
    8,  136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
    4,  132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
    12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
    2,  130, 66, 194, 34, 162,  98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
    10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
    6,  134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
    14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
    1,  129, 65, 193, 33, 161,  97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
    9,  137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
    5,  133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
    13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
    3,  131, 67, 195, 35, 163,  99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
    11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
    7,  135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
    15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255
];

#[inline(always)]
fn bit_reverse_u8(a: u8) -> u8 {
    U8_REVERSE_LOOKUP[a as usize]
}

#[inline(always)]
fn bit_reverse_u16(a: u16) -> u16 {
    (U8_REVERSE_LOOKUP[a as u8 as usize] as u16) << 8
        | U8_REVERSE_LOOKUP[(a >> 8) as u8 as usize] as u16
}

#[inline(always)]
fn bit_reverse_u32(a: u32) -> u32 {
    (U8_REVERSE_LOOKUP[a as u8 as usize] as u32) << 24
        | (U8_REVERSE_LOOKUP[(a >> 8) as u8 as usize] as u32) << 16
        | (U8_REVERSE_LOOKUP[(a >> 16) as u8 as usize] as u32) << 8
        | (U8_REVERSE_LOOKUP[(a >> 24) as u8 as usize] as u32)
}

#[inline(always)]
fn bit_reverse_u64(a: u64) -> u64 {
    (U8_REVERSE_LOOKUP[a as u8 as usize] as u64) << 56
        | (U8_REVERSE_LOOKUP[(a >> 8) as u8 as usize] as u64) << 48
        | (U8_REVERSE_LOOKUP[(a >> 16) as u8 as usize] as u64) << 40
        | (U8_REVERSE_LOOKUP[(a >> 24) as u8 as usize] as u64) << 32
        | (U8_REVERSE_LOOKUP[(a >> 32) as u8 as usize] as u64) << 24
        | (U8_REVERSE_LOOKUP[(a >> 40) as u8 as usize] as u64) << 16
        | (U8_REVERSE_LOOKUP[(a >> 48) as u8 as usize] as u64) << 8
        | (U8_REVERSE_LOOKUP[(a >> 56) as u8 as usize] as u64)
}

#[inline(always)]
pub fn bit_reverse(mut n: usize, bit_width: usize) -> usize {
    let mut right_shift: usize = 0;

    if bit_width <= 8 {
        n = bit_reverse_u8(n as u8) as usize;
        right_shift = 8 - bit_width;
    } else if bit_width <= 16 {
        n = bit_reverse_u16(n as u16) as usize;
        right_shift = 16 - bit_width;
    } else if bit_width <= 32 {
        n = bit_reverse_u32(n as u32) as usize;
        right_shift = 32 - bit_width;
    } else if bit_width <= 64 {
        n = bit_reverse_u64(n as u64) as usize;
        right_shift = 64 - bit_width;
    }

    n >> right_shift
}

#[cfg(test)]
mod bit_reverse_test {
    use crate::bit_reverse;

    #[test]
    fn test_lut_bit_reverse() {
        (1..31).for_each(|width| {
            (0..((1 << width) - 1))
                .for_each(|i| assert_eq!(bit_reverse(bit_reverse(i, width), width), i))
        })
    }
}
