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
/// impl<S, SelfInput, SelfOutput, SelfGradient> autodiff::autodiffable::ForwardDiffable<S> for T
/// where
///     Self: autodiff::autodiffable::AutoDiffable<S, Input = SelfInput, Output = SelfOutput>,
///     SelfInput: autodiff::gradienttype::GradientType<SelfOutput, GradientType = SelfGradient>,
///     SelfGradient: autodiff::forward::ForwardMul<SelfInput, SelfInput, ResultGrad = SelfOutput>,
/// {
///
///     fn eval_forward(&self, x: &Self::Input, s: &S) -> Self::Output {
///         self.eval(x, s)
///     }
///
///     fn eval_forward_grad(
///         &self,
///         x: &Self::Input,
///         dx: &Self::Input,
///         s: &S,
///     ) -> (Self::Output, Self::Output) {
///         let (f, df) = self.eval_grad(x, s);
///         (f, df.forward_mul(dx))
///     }
///
///     fn eval_forward_conj_grad(
///         &self,
///         x: &Self::Input,
///         dx: &Self::Input,
///         s: &S
///     ) -> (Self::Output, Self::Output) {
///         let (f, df) = self.eval_conj_grad(x, s);
///         (f, df.forward_mul(&dx.conj()))
///     }
///
///     fn forward_grad(
///         &self,
///         x: &Self::Input,
///         dx: &Self::Input,
///         s: &S
///     ) -> Self::Output {
///        self.grad(x, s).forward_mul(dx)
///     }
///
///     fn forward_conj_grad(
///         &self,
///         x: &Self::Input,
///         dx: &Self::Input,
///         s: &S
///     ) -> Self::Output {
///         self.conj_grad(x, s).forward_mul(&dx.conj())
///     }
/// }
/// ```
///
/// This will only work if:
/// - the struct is `AutoDiffable`
/// - the input type is `GradientType<OutputType>`
/// - the output type is `ForwardMul<InputType, InputType, ResultGrad = OutputType>`
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
            fn eval_forward(
                &self,
                x: &__PROC_MACRO_I,
                s: &__PROC_MACRO_S,
            ) -> __PROC_MACRO_O {
                self.eval(x, s)
            }

            fn eval_forward_grad(
                &self,
                x: &__PROC_MACRO_I,
                dx: &__PROC_MACRO_I,
                s: &__PROC_MACRO_S,
            ) -> (__PROC_MACRO_O, __PROC_MACRO_O) {
                let (f, df) = self.eval_grad(x, s);
                (f, df.forward_mul(dx))
            }

            fn eval_forward_conj_grad(
                &self,
                x: &__PROC_MACRO_I,
                dx: &__PROC_MACRO_I,
                s: &__PROC_MACRO_S
            ) -> (__PROC_MACRO_O, __PROC_MACRO_O) {
                let (f, df) = self.eval_conj_grad(x, s);
                (f, df.forward_mul(&dx.conj()))
            }

            fn forward_grad(
                &self,
                x: &__PROC_MACRO_I,
                dx: &__PROC_MACRO_I,
                s: &__PROC_MACRO_S
            ) -> __PROC_MACRO_O {
                self.grad(x, s).forward_mul(dx)
            }

            fn forward_conj_grad(
                &self,
                x: &__PROC_MACRO_I,
                dx: &__PROC_MACRO_I,
                s: &__PROC_MACRO_S
            ) -> __PROC_MACRO_O {
                self.conj_grad(x, s).forward_mul(&dx.conj())
            }
        }
    };

    expanded.into()
}

fn add_generics(mut generics: Generics) -> Generics {
    generics.params.push(parse_quote!(__PROC_MACRO_S));
    generics.params.push(parse_quote!(__PROC_MACRO_I));
    generics.params.push(parse_quote!(__PROC_MACRO_O));
    generics.params.push(parse_quote!(__PROC_MACRO_G));

    generics
}

fn add_bounds(mut where_clause: WhereClause) -> WhereClause {
    where_clause.predicates.push(parse_quote!(Self: autodiff::autodiffable::AutoDiffable<__PROC_MACRO_S, Input = __PROC_MACRO_I, Output = __PROC_MACRO_O>));
    where_clause.predicates.push(parse_quote!(__PROC_MACRO_I: autodiff::gradienttype::GradientType<__PROC_MACRO_O, GradientType = __PROC_MACRO_G> + autodiff::traits::Conjugate<Output = __PROC_MACRO_I>));
    where_clause.predicates.push(parse_quote!(__PROC_MACRO_G: autodiff::forward::ForwardMul<__PROC_MACRO_I, __PROC_MACRO_I, ResultGrad = __PROC_MACRO_O>));

    where_clause
}

/// derive proc macro to generate the following:
///
/// ```rust
///
/// impl<Gs..., InnerInput, InnerOutput, InnerGrad, OuterInput, OuterOutput, OuterGrad, Grad, Inner> FuncCompose<StaticArgs, Inner> for T<Gs...>
/// where
///     Self: Diffable<StaticArgs, Input = OuterInput, Output = OuterOutput>,
///     Inner: Diffable<StaticArgs, Input = InnerInput, Output = InnerOutput>,
///     OuterInput: From<InnerOutput> + GradientType<OuterOutput, GradientType = OuterGrad>,
///     InnerInput: GradientType<InnerOutput, GradientType = InnerGrad> + GradientType<OuterOutput, GradientType = Grad>,
///     OuterGrad: ForwardMul<OuterInput, InnerGrad, ResultGrad = Grad>
///
/// {
///     type Output = ADCompose<Self, Inner>;
///     fn func_compose(self, inner: Inner) -> Self::Output
///     {
///         ADCompose(self, inner)
///     }
/// }
/// ```
#[proc_macro_derive(FuncCompose)]
pub fn funccompose(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let generics_clone = input.generics.clone();
    let (_, orig_ty_generics, _) = generics_clone.split_for_impl();

    let mut generics = add_compose_generics(input.generics);
    // add a where clause if there is none
    generics.make_where_clause();

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let where_clause: Option<WhereClause> = match where_clause {
        Some(where_clause) => {
            let where_clause = where_clause.clone();
            Some(add_compose_bounds(where_clause))
        }
        None => None
    };

    let where_clause = where_clause.as_ref();

    let expanded = quote! {
        impl #impl_generics autodiff::compose::FuncCompose<StaticArgs, Inner> for #name #orig_ty_generics #where_clause {
            type Output = autodiff::adops::ADCompose<Self, Inner>;
            fn func_compose(self, inner: Inner) -> Self::Output
            {
                autodiff::adops::ADCompose(self, inner)
            }
        }
    };

    expanded.into()
}

fn add_compose_generics(mut generics: Generics) -> Generics {
    // InnerInput, InnerOutput, InnerGrad, OuterInput, OuterOutput, OuterGrad, Grad, Inner
    generics.params.push(parse_quote!(StaticArgs));
    generics.params.push(parse_quote!(InnerInput));
    generics.params.push(parse_quote!(InnerOutput));
    generics.params.push(parse_quote!(InnerGrad));
    generics.params.push(parse_quote!(OuterInput));
    generics.params.push(parse_quote!(OuterOutput));
    generics.params.push(parse_quote!(OuterGrad));
    generics.params.push(parse_quote!(Grad));
    generics.params.push(parse_quote!(Inner));

    generics
}

fn add_compose_bounds(mut where_clause: WhereClause) -> WhereClause {
    // Self: Diffable<Input = OuterInput, Output = OuterOutput>,
    // Inner: Diffable<Input = InnerInput, Output = InnerOutput>,
    // OuterInput: From<InnerOutput> + GradientType<OuterOutput, GradientType = OuterGrad>,
    // InnerInput: GradientType<InnerOutput, GradientType = InnerGrad> + GradientType<OuterOutput, GradientType = Grad>,
    // OuterGrad: ForwardMul<OuterInput, InnerGrad, ResultGrad = Grad>

    where_clause.predicates.push(parse_quote!(Self: autodiff::diffable::Diffable<StaticArgs, Input = OuterInput, Output = OuterOutput> + autodiff::autodiffable::AutoDiffable<StaticArgs> + autodiff::autodiffable::ForwardDiffable<StaticArgs>));
    where_clause.predicates.push(parse_quote!(Inner: autodiff::diffable::Diffable<StaticArgs, Input = InnerInput, Output = InnerOutput> + autodiff::autodiffable::AutoDiffable<StaticArgs> + autodiff::autodiffable::ForwardDiffable<StaticArgs>));
    where_clause.predicates.push(parse_quote!(OuterInput: From<InnerOutput> + autodiff::gradienttype::GradientType<OuterOutput, GradientType = OuterGrad>));
    where_clause.predicates.push(parse_quote!(InnerInput: autodiff::gradienttype::GradientType<InnerOutput, GradientType = InnerGrad> + autodiff::gradienttype::GradientType<OuterOutput, GradientType = Grad>));
    where_clause.predicates.push(parse_quote!(OuterGrad: autodiff::forward::ForwardMul<OuterInput, InnerGrad, ResultGrad = Grad>));

    where_clause
}
