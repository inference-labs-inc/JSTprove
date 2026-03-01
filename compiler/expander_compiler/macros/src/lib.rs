#![allow(clippy::pedantic)]
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Expr, FnArg, Ident, ItemFn, PatType, Result, ReturnType, Token, Type,
};

#[derive(Debug)]
enum ParamKind {
    Variable,
    Primitive,
}

#[derive(Debug)]
struct ParamInfo {
    name: String,
    kind: ParamKind,
    vec_depth: usize,
    is_ref: bool,
}

fn check_type_variant(ident: &Ident, allow_primitive: bool) -> Option<ParamKind> {
    if ident == "Variable" {
        Some(ParamKind::Variable)
    } else if allow_primitive
        && matches!(
            ident.to_string().as_str(),
            "u64" | "u32" | "usize" | "i64" | "i32" | "isize"
        )
    {
        Some(ParamKind::Primitive)
    } else {
        None
    }
}

fn analyze_type_structure(ty: &Type, allow_primitive: bool) -> Option<(ParamKind, usize, bool)> {
    let mut is_ref = false;
    let mut current_type = ty;

    if let Type::Reference(type_ref) = current_type {
        is_ref = true;
        current_type = type_ref.elem.as_ref();
    }

    if let Type::Path(type_path) = current_type {
        let segments = &type_path.path.segments;
        let mut vec_depth = 0;
        let mut current_segments = segments;

        if let Some(last_segment) = current_segments.last() {
            if let Some(kind) = check_type_variant(&last_segment.ident, allow_primitive) {
                return Some((kind, 0, is_ref));
            }
        }

        while let Some(last_segment) = current_segments.last() {
            if last_segment.ident == "Vec" {
                vec_depth += 1;
                if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments {
                    if let Some(syn::GenericArgument::Type(Type::Path(inner_path))) =
                        args.args.first()
                    {
                        current_segments = &inner_path.path.segments;
                        continue;
                    }
                }
                return None;
            } else {
                if let Some(kind) = check_type_variant(&last_segment.ident, allow_primitive) {
                    return Some((kind, vec_depth, is_ref));
                }
                return None;
            }
        }
    }
    None
}

#[proc_macro_attribute]
pub fn memorized(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let user_fn_name = format_ident!("{}", fn_name);
    let memorized_fn_name = format_ident!("memorized_{}", fn_name);
    let stmts = &input_fn.block.stmts;

    let generics = &input_fn.sig.generics;
    let api_arg = input_fn
        .sig
        .inputs
        .first()
        .expect("Expected at least one argument (API)");

    let api_arg_name = if let FnArg::Typed(PatType { pat, .. }) = api_arg {
        quote!(#pat).to_string().replace(" ", "")
    } else {
        panic!("API argument must be a typed parameter")
    };
    let api_arg_ident = format_ident!("{}", api_arg_name);

    let user_fn_inputs = input_fn.sig.inputs.iter().skip(1).collect::<Vec<_>>();
    let return_type = &input_fn.sig.output;

    let mut param_infos = Vec::new();
    for arg in input_fn.sig.inputs.iter().skip(1) {
        if let FnArg::Typed(PatType { pat, ty, .. }) = arg {
            let param_name = quote!(#pat).to_string().replace(" ", "");

            if let Some((kind, vec_depth, is_ref)) = analyze_type_structure(ty, true) {
                param_infos.push(ParamInfo {
                    name: param_name,
                    kind,
                    vec_depth,
                    is_ref,
                });
            } else {
                panic!("Unsupported parameter type");
            }
        }
    }

    let return_info = match &input_fn.sig.output {
        ReturnType::Type(_, ty) => {
            if let Type::Reference(_) = **ty {
                panic!("Return type cannot be a reference");
            }

            if let Some((kind, vec_depth, is_ref)) = analyze_type_structure(ty, false) {
                Some(ParamInfo {
                    name: String::from("return"),
                    kind,
                    vec_depth,
                    is_ref,
                })
            } else {
                panic!("Return type must be Vec<...Vec<Variable>...> or Variable");
            }
        }
        ReturnType::Default => None,
    };

    //eprintln!("Parameter infos: {:#?}", param_infos);
    //eprintln!("Return info: {:#?}", return_info);

    let join_variable_calls = param_infos.iter()
        .filter(|param_info| matches!(param_info.kind, ParamKind::Variable))
        .map(|param_info| {
            let param_name = format_ident!("{}", param_info.name);
            quote! {
                extra::JoinVecVariables::join_vec_variables(&#param_name, &mut inputs, &mut input_structure);
            }
        });

    let hash_primitive_calls = param_infos.iter().map(|param_info| {
        let param_name = format_ident!("{}", param_info.name);
        quote! {
            extra::HashStructureAndPrimitive::hash_structure_and_primitive(&#param_name, &mut hasher);
        }
    });

    let rebuild_vars = param_infos.iter().map(|param_info| {
        let param_name = format_ident!("{}", param_info.name);

        match param_info.kind {
            ParamKind::Variable => {
                if param_info.vec_depth > 0 {
                    let mut type_tokens = quote! { Variable };
                    for _ in 0..param_info.vec_depth {
                        type_tokens = quote! { Vec<#type_tokens> };
                    }
                    quote! {
                        let #param_name = <#type_tokens as extra::RebuildVecVariables>::rebuild_vec_variables(&mut s, &mut structure);
                    }
                } else {
                    quote! {
                        let #param_name = <Variable as extra::RebuildVecVariables>::rebuild_vec_variables(&mut s, &mut structure);
                    }
                }
            },
            ParamKind::Primitive => {
                quote! {
                    let #param_name = #param_name;
                }
            }
        }
    });

    let user_fn_args = param_infos.iter().map(|param_info| {
        let param_name = format_ident!("{}", param_info.name);
        if param_info.is_ref {
            quote! { &#param_name }
        } else {
            quote! { #param_name }
        }
    });

    let call_user_fn = quote! {
        #user_fn_name(#api_arg_ident, #(#user_fn_args),*)
    };

    let join_output = if return_info.is_some() {
        quote! {
            let result = #call_user_fn;
            let mut outputs: Vec<Variable> = Vec::new();
            let mut output_structure: Vec<usize> = Vec::new();
            extra::JoinVecVariables::join_vec_variables(&result, &mut outputs, &mut output_structure);
            #api_arg_ident.register_sub_circuit_output_structure(circuit_id, output_structure);
            outputs
        }
    } else {
        quote! {
            #call_user_fn;
            Vec::new()
        }
    };

    let rebuild_return = if let Some(return_info) = &return_info {
        let mut return_type_tokens = quote! { Variable };
        for _ in 0..return_info.vec_depth {
            return_type_tokens = quote! { Vec<#return_type_tokens> };
        }

        quote! {
            let mut s = outputs.as_slice();
            let structure = #api_arg_ident.get_sub_circuit_output_structure(circuit_id);
            let mut structure_slice = structure.as_slice();
            let result = <#return_type_tokens as extra::RebuildVecVariables>::rebuild_vec_variables(
                &mut s,
                &mut structure_slice,
            );
            result
        }
    } else {
        quote! { () }
    };

    let expanded = quote! {
        fn #user_fn_name #generics (#api_arg, #(#user_fn_inputs),*) #return_type {
            #(#stmts)*
        }

        fn #memorized_fn_name #generics (#api_arg, #(#user_fn_inputs),*) #return_type {
            use tiny_keccak::Hasher;
            let mut inputs: Vec<Variable> = Vec::new();
            let mut input_structure: Vec<usize> = Vec::new();

            #(#join_variable_calls)*

            let mut hasher = tiny_keccak::Keccak::v256();
            hasher.update(b"memorized");

            #(#hash_primitive_calls)*

            let mut hash = [0u8; 32];
            hasher.finalize(&mut hash);

            let circuit_id = #api_arg_ident.hash_to_sub_circuit_id(&hash);

            let f = |#api_arg, inputs: &Vec<Variable>| -> Vec<Variable> {
                let mut s = inputs.as_slice();
                let mut structure = input_structure.as_slice();

                #(#rebuild_vars)*

                #join_output
            };

            let outputs = #api_arg_ident.call_sub_circuit(circuit_id, &inputs, f);
            #rebuild_return
        }
    };

    //eprintln!("Expanded code: {}", expanded);
    TokenStream::from(expanded)
}

fn calculate_array_total_len(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Array(array) => {
            let len = &array.len;
            let inner_len = calculate_array_total_len(&array.elem);
            quote! { (#len as usize) * (#inner_len) }
        }
        _ => quote! { 1usize },
    }
}

fn get_array_shape(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Array(array) => {
            let len = &array.len;
            let inner_shape = get_array_shape(&array.elem);
            quote! { (#len as usize) , #inner_shape }
        }
        _ => quote! {},
    }
}

fn replace_array_with_vec(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Array(array) => {
            let elem = replace_array_with_vec(&array.elem);
            quote! { Vec<#elem> }
        }
        Type::Path(path) => {
            let ident = &path.path.segments.last().unwrap().ident;
            match ident.to_string().as_str() {
                "InputVariable" | "OutputVariable" | "InputOutputVariable" => {
                    quote! { Variable }
                }
                _ => panic!("Unsupported type: {ident}. Expected InputVariable, OutputVariable, or InputOutputVariable"),
            }
        }
        _ => panic!("Unsupported type structure. Expected array of Variable types."),
    }
}

fn get_variable_spec(ty: &Type) -> (bool, bool) {
    if let Type::Path(path) = ty {
        let ident = &path.path.segments.last().unwrap().ident;
        match ident.to_string().as_str() {
            "InputVariable" => (true, false),
            "OutputVariable" => (false, true),
            "InputOutputVariable" => (true, true),
            _ => panic!("Unsupported variable type. Expected InputVariable, OutputVariable, or InputOutputVariable"),
        }
    } else {
        panic!("Expected a Variable type")
    }
}

fn generate_array_access(vars: &[proc_macro2::Ident]) -> proc_macro2::TokenStream {
    let mut access = quote! {};
    for var in vars {
        access = quote! { #access[#var] };
    }
    access
}

fn generate_unflatten_code(
    array_index: usize,
    output_name: &proc_macro2::TokenStream,
    ty: &Type,
    is_input: bool,
) -> proc_macro2::TokenStream {
    fn collect_dimensions(ty: &Type) -> Vec<&syn::Expr> {
        let mut dims = Vec::new();
        let mut curr_ty = ty;
        while let Type::Array(array) = curr_ty {
            dims.push(&array.len);
            curr_ty = &array.elem;
        }
        dims
    }

    let dims = collect_dimensions(ty);
    if dims.is_empty() {
        if is_input {
            return quote! {
                let mut #output_name: Variable = inputs[#array_index][0];
            };
        }
        return quote! {
            let mut #output_name: Variable = Variable::default();
        };
    }

    let mut steps = Vec::with_capacity(dims.len());
    let mut step = quote! { 1 };
    for &dim in dims.iter().skip(1).rev() {
        step = quote! { #dim * (#step) };
        steps.push(step.clone());
    }
    steps.reverse();

    let loop_vars = (0..dims.len())
        .map(|i| format_ident!("i_{}", i))
        .collect::<Vec<_>>();

    let mut index_calc = quote! { 0 };
    for (i, (step, var)) in steps.iter().zip(loop_vars.iter()).enumerate() {
        if i == 0 {
            index_calc = quote! { #var * (#step) };
        } else {
            index_calc = quote! { #index_calc + #var * (#step) };
        }
    }
    if let Some(last_var) = loop_vars.last() {
        index_calc = quote! { #index_calc + #last_var };
    }

    let array_access = generate_array_access(&loop_vars[..loop_vars.len() - 1]);
    let mut inner_code = if is_input {
        quote! {
            #output_name #array_access.push(inputs[#array_index][#index_calc].clone());
        }
    } else {
        quote! {
            #output_name #array_access.push(Variable::default());
        }
    };

    for (i, (var, &dim)) in loop_vars.iter().zip(dims.iter()).enumerate().rev() {
        let init_code = if i == loop_vars.len() - 1 {
            quote! {}
        } else if i == 0 {
            quote! { #output_name.push(Vec::new()); }
        } else {
            let init_dims = &loop_vars[..i];
            let array_access = generate_array_access(init_dims);
            quote! { #output_name #array_access.push(Vec::new()); }
        };

        inner_code = quote! {
            for #var in 0..#dim {
                #init_code
                #inner_code
            }
        };
    }

    quote! {
        let mut #output_name: Vec<_> = Vec::new();
        #inner_code
    }
}

fn generate_flatten_code(
    array_index: usize,
    input_name: &proc_macro2::TokenStream,
    ty: &Type,
) -> proc_macro2::TokenStream {
    fn collect_dimensions(ty: &Type) -> Vec<&syn::Expr> {
        let mut dims = Vec::new();
        let mut curr_ty = ty;
        while let Type::Array(array) = curr_ty {
            dims.push(&array.len);
            curr_ty = &array.elem;
        }
        dims
    }

    let dims = collect_dimensions(ty);
    if dims.is_empty() {
        return quote! {
            inputs[#array_index][0] = #input_name;
        };
    }

    let mut steps = Vec::with_capacity(dims.len());
    let mut step = quote! { 1 };
    for &dim in dims.iter().skip(1).rev() {
        step = quote! { #dim * (#step) };
        steps.push(step.clone());
    }
    steps.reverse();

    let loop_vars = (0..dims.len())
        .map(|i| format_ident!("i_{}", i))
        .collect::<Vec<_>>();

    let mut index_calc = quote! { 0 };
    for (i, (step, var)) in steps.iter().zip(loop_vars.iter()).enumerate() {
        if i == 0 {
            index_calc = quote! { #var * (#step) };
        } else {
            index_calc = quote! { #index_calc + #var * (#step) };
        }
    }
    if let Some(last_var) = loop_vars.last() {
        index_calc = quote! { #index_calc + #last_var };
    }

    let array_access = generate_array_access(&loop_vars);
    let mut loop_code = quote! {
        inputs[#array_index][#index_calc] = #input_name #array_access.clone();
    };

    for (var, &dim) in loop_vars.iter().zip(dims.iter()).rev() {
        loop_code = quote! {
            for #var in 0..#dim {
                #loop_code
            }
        };
    }

    quote! { #loop_code }
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // eprintln!("Input tokens: {:#?}", item);
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let user_fn_name = format_ident!("{}", fn_name);
    let compile_fn_name = format_ident!("compile_{}", fn_name);
    let stmts = &input_fn.block.stmts;

    let mut specs = Vec::new();
    let mut shapes = Vec::new();
    let mut arg_names = Vec::new();
    let mut arg_mutability = Vec::new();
    let mut unflatten_code = Vec::new();
    let mut flatten_code = Vec::new();
    let mut current_index = 0;

    let api_arg = input_fn
        .sig
        .inputs
        .first()
        .expect("Expected at least one argument (API)");

    let user_fn_inputs: Vec<_> = input_fn
        .sig
        .inputs
        .iter()
        .skip(1)
        .map(|arg| {
            if let FnArg::Typed(PatType { pat, ty, .. }) = arg {
                if let Type::Reference(ref_type) = &**ty {
                    let vec_type = replace_array_with_vec(&ref_type.elem);
                    let arg_name = quote! { #pat };

                    let mut inner_ty = &*ref_type.elem;
                    while let Type::Array(arr) = inner_ty {
                        inner_ty = &*arr.elem;
                    }
                    let (is_input, is_output) = get_variable_spec(inner_ty);
                    let total_len = calculate_array_total_len(&ref_type.elem);
                    let shape = get_array_shape(&ref_type.elem);
                    shapes.push(quote! { vec![#shape] });

                    specs.push(quote! {
                        IOVecSpec {
                            len: #total_len,
                            is_input: #is_input,
                            is_output: #is_output,
                        }
                    });

                    arg_names.push(arg_name.clone());

                    unflatten_code.push(generate_unflatten_code(
                        current_index,
                        &arg_name,
                        &ref_type.elem,
                        is_input,
                    ));

                    if is_output {
                        flatten_code.push(generate_flatten_code(
                            current_index,
                            &arg_name,
                            &ref_type.elem,
                        ));
                    }

                    current_index += 1;

                    arg_mutability.push(ref_type.mutability.is_some());
                    if ref_type.mutability.is_some() {
                        quote! { #pat: &mut #vec_type }
                    } else {
                        quote! { #pat: &#vec_type }
                    }
                } else {
                    panic!("Expected a reference type for kernel parameters");
                }
            } else {
                panic!("Unsupported argument type for kernel function");
            }
        })
        .collect();

    let fn_args = arg_names
        .iter()
        .zip(arg_mutability.iter())
        .map(|(name, is_mut)| {
            if *is_mut {
                quote! { &mut #name }
            } else {
                quote! { &#name }
            }
        });

    let expanded = quote! {
        fn #user_fn_name<C: Config>(#api_arg, #(#user_fn_inputs),*) {
            #(#stmts)*
        }

        fn #compile_fn_name<C: Config>() -> Result<KernelPrimitive<C>, Error> {
            compile_with_spec_and_shapes(
                |api: &mut API<C>, inputs: &mut Vec<Vec<Variable>>| {
                    #(#unflatten_code)*

                    #user_fn_name(api, #(#fn_args),*);

                    #(#flatten_code)*
                },
                &[#(#specs),*],
                &[#(#shapes),*],
            )
        }
    };

    // eprintln!("Expanded tokens: {:#?}", expanded);
    // eprintln!("Expanded code: {}", expanded);
    TokenStream::from(expanded)
}

struct KernelArg {
    is_mut: bool,
    name: Ident,
}

struct KernelCall {
    ctx: Ident,
    kernel_name: Ident,
    num_parallel: Expr,
    args: Punctuated<KernelArg, Token![,]>,
}

impl Parse for KernelArg {
    fn parse(input: ParseStream) -> Result<Self> {
        let is_mut = input.peek(Token![mut]);
        if is_mut {
            input.parse::<Token![mut]>()?;
        }
        let name = input.parse()?;
        Ok(KernelArg { is_mut, name })
    }
}

impl Parse for KernelCall {
    fn parse(input: ParseStream) -> Result<Self> {
        let ctx = input.parse()?;
        input.parse::<Token![,]>()?;
        let kernel_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let num_parallel = input.parse()?;
        input.parse::<Token![,]>()?;

        let args = Punctuated::parse_terminated(input)?;

        Ok(KernelCall {
            ctx,
            kernel_name,
            num_parallel,
            args,
        })
    }
}

#[proc_macro]
pub fn call_kernel(input: TokenStream) -> TokenStream {
    let KernelCall {
        ctx,
        kernel_name,
        num_parallel,
        args,
    } = parse_macro_input!(input as KernelCall);

    // 收集所有参数名
    let arg_names: Vec<_> = args.iter().map(|arg| &arg.name).collect();

    // 分别收集可变参数的名称和索引
    let mut_vars: Vec<_> = args
        .iter()
        .enumerate()
        .filter(|(_, arg)| arg.is_mut)
        .collect();

    let mut_assignments = mut_vars.iter().map(|(i, arg)| {
        let var_name = &arg.name;
        let idx = *i;
        quote! { #var_name = io[#idx].clone(); }
    });

    // 生成代码
    let expanded = quote! {
        {
            let mut io = [#(#arg_names),*];
            let res = #ctx.call_kernel(&#kernel_name, #num_parallel, &mut io);
            #(#mut_assignments)*
            res
        }
    };

    expanded.into()
}
