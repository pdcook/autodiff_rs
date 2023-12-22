use crate::arithmetic::{
    Arithmetic, CastingArithmetic, StrongAssociatedArithmetic, WeakAssociatedArithmetic,
};

pub trait AutoDiffable<
    'a,
    StaticArgsType,
    InputType: Arithmetic<'a>,
    OutputType: WeakAssociatedArithmetic<'a, GradType>,
    GradType: StrongAssociatedArithmetic<'a, OutputType>,
> where
    &'a InputType: CastingArithmetic<'a, InputType, InputType>,
    &'a OutputType:
        CastingArithmetic<'a, OutputType, OutputType> + CastingArithmetic<'a, GradType, GradType>,
    &'a GradType:
        CastingArithmetic<'a, GradType, GradType> + CastingArithmetic<'a, OutputType, GradType>,
{
    /// required methods
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType;
    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType);

    /// provided methods
    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.eval_grad(x, static_args).1
    }
}
