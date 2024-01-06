use crate::gradienttype::GradientType;

pub trait AutoDiffable<StaticArgs> {
    type Input: GradientType<Self::Output>;
    type Output;
    /// Evaluate the function and its gradient for a given input and static arguments.
    /// Returns `(f(x, static_args): Self::Output, df/dx(x, static_args): <Self::Input as GradientType<Self::Output>>::GradientType)`
    fn eval_grad(
        &self,
        x: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, <Self::Input as GradientType<Self::Output>>::GradientType);

    /// Evaluate the function for a given input and static arguments.
    /// Returns `f(x, static_args): Self::Output`
    fn eval(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output
    {
        self.eval_grad(x, static_args).0
    }

    /// Evaluate the gradient for a given input and static arguments.
    fn grad(&self, x: &Self::Input, static_args: &StaticArgs) -> <Self::Input as GradientType<Self::Output>>::GradientType {
        self.eval_grad(x, static_args).1
    }
}

pub trait ForwardDiffable<StaticArgs> {
    type Input;
    type Output;

    /// Evaluate the function and its gradient in forward mode for a given input `x`, derivative `dx`, and static arguments
    /// Returns `(f(x, static_args): Self::Output, df(x, dx, static_args): Self::Output)`
    /// By default, `df = df/dx * dx`. However, this can be overridden in cases where this equality
    /// does not hold (e.g. complex valued functions), or where a more efficient implementation is possible (e.g. functions whose arguments and return types are arrays)
    /// NOTE: The multiplication here is not the same as normal multiplication. Instead in reality
    /// `df = (df/dx).forward_mul(dx)`. For many types, this is equivalent to normal multiplication (all primitives which implement `Mul`). However, for arrays this is tensor contraction over the last few axes, such that the number of dimensions of `df` match that of `f`.
    /// Similarly, this cannot be implemented for complex numbers, which will require a custom
    /// eval_forward_grad implementation.
    fn eval_forward_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> (Self::Output, Self::Output);

    /// Evaluate the function for a given input `x` and static arguments
    fn eval_forward(&self, x: &Self::Input, static_args: &StaticArgs) -> Self::Output {
        self.eval_forward_grad(x, x, static_args).0
    }

    /// Evaluate the gradient in forward mode for a given input `x`, derivative `dx`, and static arguments
    fn forward_grad(
        &self,
        x: &Self::Input,
        dx: &Self::Input,
        static_args: &StaticArgs,
    ) -> Self::Output {
        self.eval_forward_grad(x, dx, static_args).1
    }
}
