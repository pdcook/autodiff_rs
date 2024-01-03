pub trait AutoDiffable<StaticArgs> {
    type Input;
    type Output;
    /// Evaluate the function for a given input and static arguments.
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output;
    /// Evaluate the function and its gradient for a given input and static arguments.
    fn eval_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output);

    /// Evaluate the gradient for a given input and static arguments.
    fn grad(&self, x: &Self::Input, dx: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.eval_grad(x, dx, static_args).1
    }
}
