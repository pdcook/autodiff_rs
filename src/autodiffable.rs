pub trait AutoDiffable<StaticArgsType, InputType, OutputType, GradType, GradInputType> {
    /// Evaluate the function for a given input and static arguments.
    fn eval(&self, x: &InputType, static_args: &StaticArgsType) -> OutputType;
    /// Evaluate the function and its gradient for a given input and static arguments.
    fn eval_grad(
        &self,
        x: &InputType,
        dx: &GradInputType,
        static_args: &StaticArgsType,
    ) -> (OutputType, GradType);

    /// Evaluate the gradient for a given input and static arguments.
    fn grad(&self, x: &InputType, dx: &GradInputType, static_args: &StaticArgsType) -> GradType {
        self.eval_grad(x, dx, static_args).1
    }
}
