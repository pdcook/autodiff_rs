use crate::arithmetic::{
    Arithmetic, CastingArithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic,
};
use crate::diffable::Diffable;

pub trait AutoDiffable<
    'a,
    StaticArgsType,
    InputType: Arithmetic<'a>,
    OutputType: WeakAssociatedArithmetic<'a, GradType>,
    GradType: StrongAssociatedArithmetic<'a, OutputType>,
>: Diffable<StaticArgsType, InputType, OutputType, GradType> where
    &'a InputType: CastingArithmetic<'a, InputType, InputType>,
    &'a OutputType:
        CastingArithmetic<'a, OutputType, OutputType> + CastingArithmetic<'a, GradType, GradType>,
    &'a GradType:
        CastingArithmetic<'a, GradType, GradType> + CastingArithmetic<'a, OutputType, GradType>,
{
}
