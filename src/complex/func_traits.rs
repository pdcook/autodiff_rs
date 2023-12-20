use num::traits::Float;

pub trait ComplexFunc {
    type ReType;
    type ImType;
    type NormSqrType;
    type NormType;
    type ArgType;
    fn re(self) -> Self::ReType;
    fn im(self) -> Self::ImType;
    fn norm_sqr(self) -> Self::NormSqrType;
    fn norm(self) -> Self::NormType;
    fn arg(self) -> Self::ArgType;
}
