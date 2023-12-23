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

// implement AutoDiffable for T: Diffable with types that are arithmetic types
impl<
        'a,
        StaticArgsType,
        InputType: Arithmetic<'a>,
        OutputType: WeakAssociatedArithmetic<'a, GradType>,
        GradType: StrongAssociatedArithmetic<'a, OutputType>,
        T: Diffable<StaticArgsType, InputType, OutputType, GradType>,
    > AutoDiffable<'a, StaticArgsType, InputType, OutputType, GradType> for T
where
    &'a InputType: CastingArithmetic<'a, InputType, InputType>,
    &'a OutputType:
        CastingArithmetic<'a, OutputType, OutputType> + CastingArithmetic<'a, GradType, GradType>,
    &'a GradType:
        CastingArithmetic<'a, GradType, GradType> + CastingArithmetic<'a, OutputType, GradType>,
{
}
