pub trait AutoDiffable<StaticArgsType, InputType, OutputType, GradType> {
    /// Evaluate the function for a given input and static arguments.
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType;
    /// Evaluate the function and its gradient for a given input and static arguments.
    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType);

    /// Evaluate the gradient for a given input and static arguments.
    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.eval_grad(x, static_args).1
    }
}

pub trait CustomForwardDiff<StaticArgsType, InputType, OutputType, OutputGradType, ForwardGradType>
{
    /// Propagate the gradient forward, possibly different than the chain rule.
    fn forward_eval_grad(
        &self,
        _x: &InputType,
        _dx: Option<&ForwardGradType>,
        _static_args: &StaticArgsType,
    ) -> (OutputType, OutputGradType);

    fn forward_grad(
        &self,
        x: &InputType,
        dx: Option<&ForwardGradType>,
        static_args: &StaticArgsType,
    ) -> OutputGradType {
        self.forward_eval_grad(x, dx, static_args).1
    }
}
