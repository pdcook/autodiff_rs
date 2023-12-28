pub trait Compose<A>
where
    Self: Sized,
{
    type Output;
    fn compose(self, _other: A) -> Self::Output;

    /// alias for `compose`
    fn o(self, other: A) -> Self::Output {
        self.compose(other)
    }
    /// alias for `compose`
    fn of(self, other: A) -> Self::Output {
        self.compose(other)
    }
    /// alias for `compose`
    fn after(self, other: A) -> Self::Output {
        self.compose(other)
    }
    /// alias for `compose`
    fn on(self, other: A) -> Self::Output {
        self.compose(other)
    }
}

pub trait CustomCompose<A, OutputGradType>
where
    Self: Sized,
{
    type Output;
    fn custom_compose(self, _other: A) -> Self::Output;

    /// alias for `custom_compose`
    fn c_o(self, other: A) -> Self::Output {
        self.custom_compose(other)
    }
    /// alias for `custom_compose`
    fn c_of(self, other: A) -> Self::Output {
        self.custom_compose(other)
    }
    /// alias for `custom_compose`
    fn c_after(self, other: A) -> Self::Output {
        self.custom_compose(other)
    }
    /// alias for `custom_compose`
    fn c_on(self, other: A) -> Self::Output {
        self.custom_compose(other)
    }

    /// hack
    fn _grad_type(self, _a: OutputGradType) {
        let _ = _a;
    }
}

pub trait Abs {
    type Output;
    fn abs(self) -> Self::Output;
}

pub trait Signum {
    type Output;
    fn signum(self) -> Self::Output;
}
