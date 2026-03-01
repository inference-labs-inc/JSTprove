#![allow(clippy::pedantic)]
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, ExprPath, Ident, Token, Visibility};

struct ConfigLit {
    visibility: Visibility,
    config_name: Ident,
    field_expr: ExprPath,
    fiat_shamir_hash_type_expr: ExprPath,
    polynomial_commitment_type: ExprPath,
    scheme_config: ExprPath,
}

impl Parse for ConfigLit {
    fn parse(input: ParseStream) -> Result<Self> {
        let visibility: Visibility = input.parse()?;
        let config_name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let field_expr: ExprPath = input.parse()?;
        input.parse::<Token![,]>()?;
        let fiat_shamir_hash_type_expr: ExprPath = input.parse()?;
        input.parse::<Token![,]>()?;
        let polynomial_commitment_type: ExprPath = input.parse()?;
        input.parse::<Token![,]>()?;
        let scheme_config: ExprPath = input.parse()?;
        let _ = input.parse::<Token![,]>();
        Ok(ConfigLit {
            visibility,
            config_name,
            field_expr,
            fiat_shamir_hash_type_expr,
            polynomial_commitment_type,
            scheme_config,
        })
    }
}

fn parse_field_type(field_expr: ExprPath) -> (String, String) {
    let field_enum = field_expr
        .path
        .segments
        .last()
        .expect("Empty path for field");
    match field_enum.ident.to_string().as_str() {
        "M31x1" => ("M31x1".to_owned(), "M31x1Config".to_owned()),
        "M31x16" => ("M31x16".to_owned(), "M31x16Config".to_owned()),
        "BN254" => ("BN254".to_owned(), "BN254Config".to_owned()),
        "GF2Ext128" => ("GF2Ext128".to_owned(), "GF2ExtConfig".to_owned()),
        "Goldilocksx1" => ("Goldilocksx1".to_owned(), "Goldilocksx1Config".to_owned()),
        "Goldilocksx8" => ("Goldilocksx8".to_owned(), "Goldilocksx8Config".to_owned()),
        "BabyBearx16" => ("BabyBearx16".to_owned(), "BabyBearx16Config".to_owned()),
        _ => panic!("Unknown field type"),
    }
}

fn parse_fiat_shamir_hash_type(
    field_type: &str,
    field_config: &str,
    fiat_shamir_hash_type: ExprPath,
) -> (String, String) {
    let hash_enum = fiat_shamir_hash_type
        .path
        .segments
        .last()
        .expect("Empty path for hash type");

    let binding = hash_enum.ident.to_string();
    let hash_type_str = binding.as_str();
    let challenge_f = format!("<{field_config} as FieldEngine>::ChallengeField");
    match (hash_type_str, field_type) {
        ("SHA256", _) => (
            "SHA256".to_owned(),
            "BytesHashTranscript::<SHA256hasher>".to_owned(),
        ),
        ("Keccak256", _) => (
            "Keccak256".to_owned(),
            "BytesHashTranscript::<Keccak256hasher>".to_owned(),
        ),
        ("Poseidon", "M31x16") => (
            "Poseidon".to_owned(),
            "BytesHashTranscript::<PoseidonFiatShamirHasher<M31x16>>".to_owned(),
        ),
        ("MIMC5", "BN254") => (
            "MIMC5".to_owned(),
            format!("BytesHashTranscript::<MiMC5FiatShamirHasher<{challenge_f}>>").to_owned(),
        ),
        _ => panic!("Unknown hash type"),
    }
}

fn parse_polynomial_commitment_type(
    field_type: &str,
    field_config: &str,
    polynomial_commitment_type: ExprPath,
) -> (String, String) {
    let binding = polynomial_commitment_type
        .path
        .segments
        .last()
        .expect("Empty path for polynomial commitment type");

    let pcs_type_str = binding.ident.to_string();
    match (pcs_type_str.as_str(), field_type) {
        ("Raw", _) => (
            "Raw".to_owned(),
            format!("RawExpanderGKR::<{field_config}>").to_owned(),
        ),
        ("Hyrax", "BN254") => ("Hyrax".to_string(), "HyraxPCS::<G1Affine>".to_string()),
        ("KZG", "BN254") => ("KZG".to_owned(), "HyperBiKZGPCS::<Bn256>".to_string()),
        ("Orion", "GF2Ext128") => (
            "Orion".to_owned(),
            format!("OrionPCSForGKR::<{field_config}, GF2x128>").to_owned(),
        ),
        ("Orion", "M31x16") => (
            "Orion".to_owned(),
            format!("OrionPCSForGKR::<{field_config}, M31x16>").to_owned(),
        ),
        ("Orion", "Goldilocksx8") => (
            "Orion".to_owned(),
            format!("OrionPCSForGKR::<{field_config}, Goldilocksx8>").to_owned(),
        ),
        _ => panic!(
            "Unknown polynomial commitment type in config macro expansion. PCS: '{pcs_type_str}', Field: '{field_type}'"
        ),
    }
}

fn _parse_scheme_config(scheme_config: ExprPath) -> String {
    let binding = scheme_config
        .path
        .segments
        .last()
        .expect("Empty path for scheme config");
    binding.ident.to_string()
}

#[proc_macro]
pub fn declare_gkr_config(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    declare_gkr_config_impl(input)
}

fn declare_gkr_config_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ConfigLit {
        visibility,
        config_name,
        field_expr,
        fiat_shamir_hash_type_expr,
        polynomial_commitment_type,
        scheme_config,
    } = parse_macro_input!(input as ConfigLit);

    let (field_type, field_config) = parse_field_type(field_expr);
    let (_fiat_shamir_hash_type, transcript_type) =
        parse_fiat_shamir_hash_type(&field_type, &field_config, fiat_shamir_hash_type_expr);
    let (_polynomial_commitment_enum, polynomial_commitment_type) =
        parse_polynomial_commitment_type(&field_type, &field_config, polynomial_commitment_type);

    let field_config = format_ident!("{field_config}");
    let transcript_type_expr = syn::parse_str::<syn::Type>(&transcript_type).unwrap();
    let polynomial_commitment_type_expr =
        syn::parse_str::<syn::Type>(&polynomial_commitment_type).unwrap();

    let ret: TokenStream = quote! {
        #[derive(Default, Debug, Clone, PartialOrd, Ord, Hash, PartialEq, Eq, Copy)]
        #visibility struct #config_name;

        impl GKREngine for #config_name {
            type FieldConfig = #field_config;
            type MPIConfig = MPIConfig;
            type TranscriptConfig = #transcript_type_expr;
            type PCSConfig = #polynomial_commitment_type_expr;
            const SCHEME: GKRScheme = #scheme_config;
        }
    };

    ret.into()
}
