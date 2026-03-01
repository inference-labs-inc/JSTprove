#![allow(clippy::pedantic)]
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(ExpSerde)]
pub fn serdes_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let expanded = match input.data {
        Data::Struct(data) => {
            let fields = match data.fields {
                Fields::Named(fields) => fields.named,
                Fields::Unnamed(fields) => fields.unnamed,
                Fields::Unit => return quote! {}.into(),
            };

            let field_names: Vec<_> = fields
                .iter()
                .map(|field| {
                    field.ident.as_ref().unwrap_or_else(|| {
                        panic!("Tuple structs are not supported by ExpSerde derive macro")
                    })
                })
                .collect();

            let field_types: Vec<_> = fields.iter().map(|field| &field.ty).collect();

            let serialize_impl = quote! {
                fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> ::serdes::SerdeResult<()> {
                    #(
                        self.#field_names.serialize_into(&mut writer)?;
                    )*
                    Ok(())
                }
            };

            let deserialize_impl = quote! {
                fn deserialize_from<R: std::io::Read>(mut reader: R) -> ::serdes::SerdeResult<Self> {
                    Ok(Self {
                        #(
                            #field_names: <#field_types as ::serdes::ExpSerde>::deserialize_from(&mut reader)?,
                        )*
                    })
                }
            };

            quote! {
                impl #impl_generics ::serdes::ExpSerde for #name #ty_generics #where_clause {
                    #serialize_impl
                    #deserialize_impl
                }
            }
        }
        Data::Enum(data) => {
            let variants = data.variants;
            let _variant_count = variants.len();
            let mut serialize_arms = Vec::new();
            let mut deserialize_arms = Vec::new();
            for (idx, variant) in variants.iter().enumerate() {
                let vname = &variant.ident;
                match &variant.fields {
                    Fields::Unit => {
                        serialize_arms.push(quote! {
                            Self::#vname => {
                                (#idx as u32).serialize_into(&mut writer)?;
                            }
                        });
                        deserialize_arms.push(quote! {
                            #idx => Ok(Self::#vname),
                        });
                    }
                    Fields::Unnamed(fields) => {
                        let field_idents: Vec<_> = (0..fields.unnamed.len())
                            .map(|i| syn::Ident::new(&format!("f{i}"), vname.span()))
                            .collect();
                        serialize_arms.push(quote! {
                            Self::#vname( #( ref #field_idents ),* ) => {
                                (#idx as u32).serialize_into(&mut writer)?;
                                #(
                                    #field_idents.serialize_into(&mut writer)?;
                                )*
                            }
                        });
                        let deser_fields = field_idents.iter().map(
                            |_| quote! { <_ as ::serdes::ExpSerde>::deserialize_from(&mut reader)? },
                        );
                        deserialize_arms.push(quote! {
                            #idx => Ok(Self::#vname( #(#deser_fields),* )),
                        });
                    }
                    Fields::Named(fields) => {
                        let field_idents: Vec<_> = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect();
                        serialize_arms.push(quote! {
                            Self::#vname { #( ref #field_idents ),* } => {
                                (#idx as u32).serialize_into(&mut writer)?;
                                #(
                                    #field_idents.serialize_into(&mut writer)?;
                                )*
                            }
                        });
                        let deser_fields = field_idents.iter().map(|ident| quote! { #ident: <_ as ::serdes::ExpSerde>::deserialize_from(&mut reader)? });
                        deserialize_arms.push(quote! {
                            #idx => Ok(Self::#vname { #(#deser_fields),* }),
                        });
                    }
                }
            }
            let serialize_variant = quote! {
                fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> ::serdes::SerdeResult<()> {
                    match self {
                        #(#serialize_arms)*
                    }
                    Ok(())
                }
            };
            let deserialize_variant = quote! {
                fn deserialize_from<R: std::io::Read>(mut reader: R) -> ::serdes::SerdeResult<Self> {
                    let variant_index: u32 = ::serdes::ExpSerde::deserialize_from(&mut reader)?;
                    match variant_index as usize {
                        #(#deserialize_arms)*
                        _ => Err(::serdes::SerdeError::InvalidVariantIndex(variant_index as usize)),
                    }
                }
            };
            quote! {
                impl #impl_generics ::serdes::ExpSerde for #name #ty_generics #where_clause {
                    #serialize_variant
                    #deserialize_variant
                }
            }
        }
        Data::Union(_) => {
            quote! {
                compile_error!("Unions are not supported by ExpSerde derive macro");
            }
        }
    };

    expanded.into()
}
