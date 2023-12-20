pub trait Compose<A> {
    type Output;
    fn compose(self, _other: A) -> Self::Output;
}

pub trait Signed {
    type AbsType;
    type SignType;
    fn abs(self) -> Self::AbsType;
    fn signum(self) -> Self::SignType;
}
