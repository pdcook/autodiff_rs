use ndarray::{Dimension, DimMax, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

pub trait DimSub<Rhs>
where
    Rhs: Dimension,
    Self: Dimension + DimMax<Rhs, Output = Self>,
{
    type Output;
}

// sub from IxDyn
impl<D> DimSub<D> for IxDyn
where
    D: Dimension,
    Self: Dimension + DimMax<D, Output = IxDyn>,
{
    type Output = IxDyn;
}

macro_rules! impl_dim_sub {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl DimSub<$rhs> for $lhs {
            type Output = $out;
        }
    };
}

// impl (In, Sub, Out)
impl_dim_sub!(Ix0, Ix0, Ix0);
impl_dim_sub!(Ix1, Ix0, Ix1);
impl_dim_sub!(Ix1, Ix1, Ix0);
impl_dim_sub!(Ix2, Ix0, Ix2);
impl_dim_sub!(Ix2, Ix1, Ix1);
impl_dim_sub!(Ix2, Ix2, Ix0);
impl_dim_sub!(Ix3, Ix0, Ix3);
impl_dim_sub!(Ix3, Ix1, Ix2);
impl_dim_sub!(Ix3, Ix2, Ix1);
impl_dim_sub!(Ix3, Ix3, Ix0);
impl_dim_sub!(Ix4, Ix0, Ix4);
impl_dim_sub!(Ix4, Ix1, Ix3);
impl_dim_sub!(Ix4, Ix2, Ix2);
impl_dim_sub!(Ix4, Ix3, Ix1);
impl_dim_sub!(Ix4, Ix4, Ix0);
impl_dim_sub!(Ix5, Ix0, Ix5);
impl_dim_sub!(Ix5, Ix1, Ix4);
impl_dim_sub!(Ix5, Ix2, Ix3);
impl_dim_sub!(Ix5, Ix3, Ix2);
impl_dim_sub!(Ix5, Ix4, Ix1);
impl_dim_sub!(Ix5, Ix5, Ix0);
impl_dim_sub!(Ix6, Ix0, Ix6);
impl_dim_sub!(Ix6, Ix1, Ix5);
impl_dim_sub!(Ix6, Ix2, Ix4);
impl_dim_sub!(Ix6, Ix3, Ix3);
impl_dim_sub!(Ix6, Ix4, Ix2);
impl_dim_sub!(Ix6, Ix5, Ix1);
impl_dim_sub!(Ix6, Ix6, Ix0);
