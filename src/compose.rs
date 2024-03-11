pub trait AutoCompose<A>
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

pub trait FuncCompose<StaticArgs, Other> {
    type Output;

    // default implementation via proc macro uses `ADCompose`
    fn func_compose(self, other: Other) -> Self::Output;
}
