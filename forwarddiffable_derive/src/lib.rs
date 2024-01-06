extern crate quote;
extern crate syn;

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse_quote, DeriveInput, WhereClause, Generics};

/// implement a default ForwardDiffable with trait bounds as the following:
///
/// ```rust
///
/// impl<__PROC_MACRO_S, __PROC_MACRO_I, __PROC_MACRO_O> autodiff::autodiffable::ForwardDiffable<__PROC_MACRO_S> for T
/// where
///     Self: autodiff::autodiffable::AutoDiffable<__PROC_MACRO_S, Input = __PROC_MACRO_I, Output = __PROC_MACRO_O>,
///     __PROC_MACRO_I: Clone + autodiff::gradienttype::GradientType<__PROC_MACRO_O, GradientType = __PROC_MACRO_O>,
///     __PROC_MACRO_O: autodiff::forward::ForwardMul<__PROC_MACRO_I, __PROC_MACRO_O, __PROC_MACRO_I>,
/// {
///     fn eval_forward_grad(
///         &self,
///         x: &__PROC_MACRO_I,
///         dx: &__PROC_MACRO_I,
///         s: &__PROC_MACRO_S,
///     ) -> (__PROC_MACRO_O, __PROC_MACRO_O) {
///         let (f, df) = self.eval_grad(x, s);
///         (f, df.forward_mul(dx.clone()))
///     }
///
/// }
/// ```
///
/// This will only work if:
/// - the struct is AutoDiffable
/// - the struct's AutoDiffable function's output type and gradient type are equal
/// - the input type is Clone and GradientType<OutputType, GradientType = OutputType>
/// - the output type is ForwardMul<InputType, OutputType, InputType>
///
#[proc_macro_derive(SimpleForwardDiffable)]
pub fn simple_forward_diffable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let generics_clone = input.generics.clone();
    let (_, orig_ty_generics, _) = generics_clone.split_for_impl();

    let mut generics = add_generics(input.generics);
    // add a where clause if there is none
    generics.make_where_clause();

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let where_clause: Option<WhereClause> = match where_clause {
        Some(where_clause) => {
            let where_clause = where_clause.clone();
            Some(add_bounds(where_clause))
        }
        None => None
    };

    let where_clause = where_clause.as_ref();

    let expanded = quote! {
        impl #impl_generics autodiff::autodiffable::ForwardDiffable<__PROC_MACRO_S> for #name #orig_ty_generics #where_clause {
            fn eval_forward_grad(
                &self,
                x: &__PROC_MACRO_I,
                dx: &__PROC_MACRO_I,
                s: &__PROC_MACRO_S,
            ) -> (__PROC_MACRO_O, __PROC_MACRO_O) {
                let (f, df) = self.eval_grad(x, s);
                (f, df.forward_mul(dx.clone()))
            }
        }
    };

    expanded.into()
}

fn add_generics(mut generics: Generics) -> Generics {
    generics.params.push(parse_quote!(__PROC_MACRO_S));
    generics.params.push(parse_quote!(__PROC_MACRO_I));
    generics.params.push(parse_quote!(__PROC_MACRO_O));

    generics
}

fn add_bounds(mut where_clause: WhereClause) -> WhereClause {
    where_clause.predicates.push(parse_quote!(Self: autodiff::autodiffable::AutoDiffable<__PROC_MACRO_S, Input = __PROC_MACRO_I, Output = __PROC_MACRO_O>));
    where_clause.predicates.push(parse_quote!(__PROC_MACRO_I: Clone + autodiff::gradienttype::GradientType<__PROC_MACRO_O, GradientType = __PROC_MACRO_O>));
    where_clause.predicates.push(parse_quote!(__PROC_MACRO_O: std::ops::Mul<__PROC_MACRO_I, Output = __PROC_MACRO_O> + autodiff::forward::ForwardMul<__PROC_MACRO_I, __PROC_MACRO_O, __PROC_MACRO_I>));

    where_clause
}
