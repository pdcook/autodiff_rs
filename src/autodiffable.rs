pub trait AutoDiffable<StaticArgsType, InputType, OutputType, GradType> {
    /// Evaluate the function for a given input and static arguments.
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType;
    /// Evaluate the function and its gradient for a given input and static arguments.
    fn eval_grad(&self, x: &InputType, static_args: &StaticArgsType) -> (OutputType, GradType);

    /// Propagate the gradient forward, possibly different than the chain rule.
    fn forward_eval_grad<ForwardGradType, OutputGradType>(
        &self,
        _x: &InputType,
        _dx: Option<&ForwardGradType>,
        _static_args: &StaticArgsType,
    ) -> Result<(OutputType, OutputGradType), &'static str>
    {
        Err("Forward mode not implemented for this function")
    }
    /// Evaluate the gradient for a given input and static arguments.
    fn grad(&self, x: &InputType, static_args: &StaticArgsType) -> GradType {
        self.eval_grad(x, static_args).1
    }

    fn forward_grad<ForwardGradType, OutputGradType>(
        &self,
        x: &InputType,
        dx: Option<&ForwardGradType>,
        static_args: &StaticArgsType,
    ) -> Result<OutputGradType, &'static str>
    {
        self.forward_eval_grad(x, dx, static_args).map(|(_, grad)| grad)
    }
}
