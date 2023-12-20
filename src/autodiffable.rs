use crate::arithmetic::{Arithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic};

pub trait AutoDiffable<StaticArgsType> {
    type InType: Arithmetic;
    type OutType: WeakAssociatedArithmetic<Self::GradType>;
    type GradType: StrongAssociatedArithmetic<Self::OutType>;

    fn eval(&self, x: Self::InType, static_args: &StaticArgsType) -> Self::OutType;
    fn grad(&self, x: Self::InType, static_args: &StaticArgsType) -> Self::GradType;
}
