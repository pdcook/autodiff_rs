use crate::arithmetic::{
    Arithmetic, CastingArithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic,
};

pub trait AutoDiffable<StaticArgsType> {
    type InType: Arithmetic
    where
        for<'a> &'a Self::InType: CastingArithmetic<Self::InType, Self::InType>;
    type OutType: WeakAssociatedArithmetic<Self::GradType>
    where
        for<'a> &'a Self::OutType: CastingArithmetic<Self::OutType, Self::OutType>
            + CastingArithmetic<Self::GradType, Self::GradType>;
    type GradType: StrongAssociatedArithmetic<Self::OutType>
    where
        for<'a> &'a Self::GradType: CastingArithmetic<Self::GradType, Self::GradType>
            + CastingArithmetic<Self::OutType, Self::GradType>;

    fn eval(&self, x: &Self::InType, static_args: &StaticArgsType) -> Self::OutType;
    fn grad(&self, x: &Self::InType, static_args: &StaticArgsType) -> Self::GradType;
}
