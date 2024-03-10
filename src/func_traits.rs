/*pub trait Compose<A>
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
}*/

// forward from traits
pub use crate::traits::{InstOne, InstZero, Wirtinger};

pub trait Abs {
    type Output;
    fn abs(self) -> Self::Output;
}

pub trait Signum {
    type Output;
    fn signum(self) -> Self::Output;
}
