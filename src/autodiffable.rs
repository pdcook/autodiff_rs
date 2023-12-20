use crate::arithmetic::{
    Arithmetic, CastingArithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic,
};

pub trait AutoDiffable<
    StaticArgsType,
    InputType: Arithmetic,
    OutputType: WeakAssociatedArithmetic<GradType>,
    GradType: StrongAssociatedArithmetic<OutputType>,
> where
    for<'a> &'a InputType: CastingArithmetic<InputType, InputType>,
    for<'a> &'a OutputType:
        CastingArithmetic<OutputType, OutputType> + CastingArithmetic<GradType, GradType>,
    for<'a> &'a GradType:
        CastingArithmetic<GradType, GradType> + CastingArithmetic<OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType;
    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType;
}
