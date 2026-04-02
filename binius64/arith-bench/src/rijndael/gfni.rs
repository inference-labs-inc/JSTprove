// Copyright 2025 Irreducible Inc.

use crate::underlier::{OpsGfni, PackedUnderlier, Underlier};

pub fn mul<U: Underlier + OpsGfni>(a: U, b: U) -> U {
    OpsGfni::gf2p8mul(a, b)
}

pub fn sqr<U: Underlier + OpsGfni>(x: U) -> U {
    OpsGfni::gf2p8mul(x, x)
}

pub fn inv<U: Underlier + OpsGfni + PackedUnderlier<u64>>(x: U) -> U {
    #[rustfmt::skip]
	pub const IDENTITY_MAP: u64 = u64::from_le_bytes([
		0b10000000,
		0b01000000,
		0b00100000,
		0b00010000,
		0b00001000,
		0b00000100,
		0b00000010,
		0b00000001,
	]);

    let identity_map = <U as PackedUnderlier<u64>>::broadcast(IDENTITY_MAP);
    OpsGfni::gf2p8affineinv::<0>(x, identity_map)
}
