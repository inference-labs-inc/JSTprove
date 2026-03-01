use crate::gnark::element::{value_of, Element};
use crate::gnark::emparam::{CurveParams, FieldParams};
use crate::gnark::field::GField;
use expander_compiler::frontend::{Config, RootAPI};
use num_bigint::BigInt;

#[derive(Default, Clone)]
pub struct AffinePoint<Base: FieldParams> {
    pub x: Element<Base>,
    pub y: Element<Base>,
}

pub struct Curve<Base: FieldParams, Scalars: FieldParams> {
    pub params: CurveParams,
    // baseApi is the api for point operations on the base field.
    pub base_api: GField<Base>,
    // pub scalar_api: EmulatedField<Scalars>,
    // g is the generator (base point) of the curve.
    pub g: AffinePoint<Base>,
    // gm are the pre-computed doubles the generator (base point) of the curve.
    // pub gm: Vec<AffinePoint<Base>>,
    pub a: Element<Base>,
    pub b: Element<Base>,
    // indicates whether the curve has a non-zero A parameter.
    pub add_a: bool,
    // pub eigenvalue: Option<Scalars>,
    // pub third_root_one: Option<Base>,
    _maker: std::marker::PhantomData<Scalars>,
}

impl<Base: FieldParams, Scalars: FieldParams> Curve<Base, Scalars> {
    pub fn new<C: Config, B: RootAPI<C>>(
        native: &mut B,
        params: &CurveParams,
        fparam: Base,
    ) -> Self {
        let base_api = GField::new(native, fparam);

        let gx = value_of::<C, B, Base>(native, Box::new(params.gx.clone()));
        let gy = value_of::<C, B, Base>(native, Box::new(params.gy.clone()));

        let (a, b) = (params.a.clone(), params.b.clone());
        Self {
            params: params.clone(),
            base_api,
            // scalar_api,
            g: AffinePoint { x: gx, y: gy },
            // gm: emu_gm,
            a: value_of::<C, B, Base>(native, Box::new(a)),
            b: value_of::<C, B, Base>(native, Box::new(b)),
            add_a: params.a != BigInt::ZERO,
            // eigenvalue,
            // third_root_one,
            _maker: std::marker::PhantomData,
        }
    }

    pub fn generator(&self) -> &AffinePoint<Base> {
        &self.g
    }

    pub fn assert_is_on_curve<C: Config, B: RootAPI<C>>(
        &mut self,
        builder: &mut B,
        p: &AffinePoint<Base>,
    ) {
        let x_is_zero = self.base_api.is_zero::<C, B>(builder, &p.x);
        let y_is_zero = self.base_api.is_zero::<C, B>(builder, &p.y);
        let selector = builder.and(x_is_zero, y_is_zero);
        let zero_const = self.base_api.zero_const.clone();
        let b = self
            .base_api
            .select(builder, selector, &zero_const, &self.b);

        let left = self.base_api.mul(builder, &p.y, &p.y);
        let px_squared = self.base_api.mul(builder, &p.x, &p.x);
        let mut right = self.base_api.mul(builder, &p.x, &px_squared);
        right = self.base_api.add(builder, &right, &b);

        if self.add_a {
            let ax = self.base_api.mul(builder, &self.a, &p.x);
            right = self.base_api.add(builder, &right, &ax);
        }

        self.base_api.assert_is_equal(builder, &left, &right);
    }

    // Neg returns an inverse of p. It doesn't modify p.
    pub fn neg<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &AffinePoint<Base>,
    ) -> AffinePoint<Base> {
        AffinePoint {
            x: p.x.clone(),
            y: self.base_api.neg(native, &p.y),
        }
    }
}
