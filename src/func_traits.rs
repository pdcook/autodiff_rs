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

pub trait Signed {
    type AbsType;
    type SignType;
    fn abs(self) -> Self::AbsType;
    fn signum(self) -> Self::SignType;
}
