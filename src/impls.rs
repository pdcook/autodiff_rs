use crate::autodiffable::AutoDiffable;
use crate::traits::InstZero;

// implementation of AutoDiffable for constants that implement InstZero and Clone

impl<S, I, T: InstZero + Clone> AutoDiffable<S, I, T, T, I> for T {
    fn eval(&self, _: &I, _: &S) -> T {
        self.clone()
    }
    fn eval_grad(&self, _: &I, _: &I, _: &S) -> (T, T) {
        let grad = self.zero();
        (self.clone(), grad)
    }
}
