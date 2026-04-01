pub mod inverse;
pub mod jacobi;
use subtle::{Choice, ConstantTimeEq};

pub trait Legendre {
    fn legendre(&self) -> i64;

    #[inline(always)]
    fn ct_quadratic_non_residue(&self) -> Choice {
        self.legendre().ct_eq(&-1)
    }

    #[inline(always)]
    fn ct_quadratic_residue(&self) -> Choice {
        // The legendre symbol returns 0 for 0
        // and 1 for quadratic residues,
        // we consider 0 a square hence quadratic residue.
        self.legendre().ct_ne(&-1)
    }
}

#[macro_export]
macro_rules! extend_field_legendre {
    ($field:ident ) => {
        impl $crate::ff_ext::Legendre for $field {
            #[inline(always)]
            fn legendre(&self) -> i64 {
                self.jacobi()
            }
        }
    };
}
