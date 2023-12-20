use crate::arithmetic::*;
use num::complex::Complex;
use num::traits::Float;

// num crate does not provide a trait for Complex numbers
// only a struct. This trait is used to provide a common
// trait for all complex numbers
pub trait ComplexTrait<F: Float> {
    fn new(re: F, im: F) -> Self;
    fn i() -> Self;
    fn re(&self) -> Self;
    fn im(&self) -> Self;
    fn conj(&self) -> Self;
    fn norm_sqr(&self) -> Self;
    fn norm(&self) -> Self;
    fn arg(&self) -> Self;
}

// Implemented for all Complex<Float> types
impl<F: Float> ComplexTrait<F> for Complex<F> {
    fn new(re: F, im: F) -> Self {
        Complex::new(re, im)
    }
    fn i() -> Self {
        Complex::i()
    }
    fn re(&self) -> Self {
        Complex::new(self.re, F::zero())
    }
    fn im(&self) -> Self {
        Complex::new(F::zero(), self.im)
    }
    fn conj(&self) -> Self {
        Complex::conj(self)
    }
    fn norm_sqr(&self) -> Self {
        Complex::new(Complex::norm_sqr(self), F::zero())
    }
    fn norm(&self) -> Self {
        Complex::new(Complex::norm(*self), F::zero())
    }
    fn arg(&self) -> Self {
        Complex::new(Complex::arg(*self), F::zero())
    }
}

// supertrait for complex arithmetic
pub trait ComplexArithmetic<F: Float>: Arithmetic + ComplexTrait<F> {}

// supertrait for strong associated complex arithmetic
pub trait ComplexStrongAssociatedArithmetic<F: Float, FT: Float, T: ComplexArithmetic<FT>>:
    StrongAssociatedArithmetic<T> + ComplexTrait<F>
{
}

// supertrait for weak associated complex arithmetic
pub trait ComplexWeakAssociatedArithmetic<F: Float, FT: Float, T: ComplexArithmetic<FT>>:
    WeakAssociatedArithmetic<T> + ComplexTrait<F>
{
}

// supertrait for extended complex arithmetic
pub trait ComplexExtendedArithmetic<F: Float>: ExtendedArithmetic + ComplexTrait<F> {}

// supertrait for extended strong associated complex arithmetic
pub trait ComplexStrongAssociatedExtendedArithmetic<
    F: Float,
    FT: Float,
    T: ComplexExtendedArithmetic<FT>,
>: StrongAssociatedExtendedArithmetic<T> + ComplexTrait<F>
{
}

// supertrait for extended weak associated complex arithmetic
pub trait ComplexWeakAssociatedExtendedArithmetic<
    F: Float,
    FT: Float,
    T: ComplexExtendedArithmetic<FT>,
>: WeakAssociatedExtendedArithmetic<T> + ComplexTrait<F>
{
}

// blanket implementations for all types
// so they can be used like a trait alias
impl<F: Float, T: Arithmetic + ComplexTrait<F>> ComplexArithmetic<F> for T {}

impl<
        FT: Float,
        FU: Float,
        T: ComplexArithmetic<FT> + StrongAssociatedArithmetic<U>,
        U: ComplexArithmetic<FU>,
    > ComplexStrongAssociatedArithmetic<FT, FU, U> for T
{
}

impl<
        FT: Float,
        FU: Float,
        T: ComplexArithmetic<FT> + WeakAssociatedArithmetic<U>,
        U: ComplexArithmetic<FU>,
    > ComplexWeakAssociatedArithmetic<FT, FU, U> for T
{
}

impl<F: Float, T: ExtendedArithmetic + ComplexTrait<F>> ComplexExtendedArithmetic<F> for T {}

impl<
        FT: Float,
        FU: Float,
        T: ComplexExtendedArithmetic<FT> + StrongAssociatedExtendedArithmetic<U>,
        U: ComplexExtendedArithmetic<FU>,
    > ComplexStrongAssociatedExtendedArithmetic<FT, FU, U> for T
{
}

impl<
        FT: Float,
        FU: Float,
        T: ComplexExtendedArithmetic<FT> + WeakAssociatedExtendedArithmetic<U>,
        U: ComplexExtendedArithmetic<FU>,
    > ComplexWeakAssociatedExtendedArithmetic<FT, FU, U> for T
{
}
