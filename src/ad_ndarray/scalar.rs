use ndarray::Array0;

pub type Scalar<T> = Array0<T>;

pub trait ScalarOps<T: Clone + Copy> {
    fn new(value: T) -> Self;
    fn value(&self) -> T;
    fn set_value(&mut self, value: T);
}

impl<T: Clone + Copy> ScalarOps<T> for Scalar<T> {
    fn new(value: T) -> Self {
        Array0::from_elem((), value)
    }
    fn value(&self) -> T {
        self[()]
    }
    fn set_value(&mut self, value: T) {
        self[()] = value;
    }
}
