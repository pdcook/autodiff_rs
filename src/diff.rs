use crate::diffable::Diffable;
use std::marker::PhantomData;
use std::ops::Deref;

/// A wrapper type for an Diffable type.
#[derive(Debug, Clone)]
pub struct Diff<StaticArgsType, InputType, OutputType, GradType, T>(
    pub T,
    pub PhantomData<(StaticArgsType, InputType, OutputType, GradType)>,
)
where
    T: Diffable<StaticArgsType, InputType, OutputType, GradType>;

/// Impl Deref for Diff
impl<StaticArgsType, InputType, OutputType, GradType, T> Deref
    for Diff<StaticArgsType, InputType, OutputType, GradType, T>
where
    T: Diffable<StaticArgsType, InputType, OutputType, GradType>,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Impl Copy for Diff if T is Copy and all the other types are Clone
impl<StaticArgsType, InputType, OutputType, GradType, T> Copy
    for Diff<StaticArgsType, InputType, OutputType, GradType, T>
where
    StaticArgsType: Clone,
    InputType: Clone,
    OutputType: Clone,
    GradType: Clone,
    T: Copy + Diffable<StaticArgsType, InputType, OutputType, GradType>,
{
}

/// Impl of new for Diff
impl<StaticArgsType, InputType, OutputType, GradType, T>
    Diff<StaticArgsType, InputType, OutputType, GradType, T>
where
    T: Diffable<StaticArgsType, InputType, OutputType, GradType>,
{
    #[allow(dead_code)]
    pub fn new(t: T) -> Self {
        Diff(t, PhantomData)
    }
}

/// Impl of Diffable for Diff
impl<StaticArgsType, InputType, OutputType, GradType, T>
    Diffable<StaticArgsType, InputType, OutputType, GradType>
    for Diff<StaticArgsType, InputType, OutputType, GradType, T>
where
    T: Diffable<StaticArgsType, InputType, OutputType, GradType>,
{
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType {
        self.0.eval(x, static_args)
    }

    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType) {
        self.0.eval_grad(x, static_args)
    }
}
