use crate::ad_ndarray::dimabssub::DimAbsSub;
use ndarray::{ArrayBase, Axis, Data, Dim, DimAdd, DimMax, Dimension, LinalgScalar, OwnedRepr};
use ndarray_einsum_beta;

pub use ndarray::linalg::Dot; // re-export ndarray::linalg::Dot
use ndarray_linalg;

/// N-dimensional dot product trait
/// performs contraction of the last axis of the first array and the first axis of the second array
pub trait TensorDot<T> {
    type Output;
    fn tensordot(&self, other: &T) -> Self::Output;
}

impl<
        A,    // element type, must be the same for both arrays and the result
        SS,   // array storage for self
        DS,   // self dimension
        SO,   // other storage
        DO,   // other dimension
        DR,   // result dimension
        DSUM, // sum of self and other dimension
    > TensorDot<ArrayBase<SO, DO>> for ArrayBase<SS, DS>
where
    A: LinalgScalar,
    SS: Data<Elem = A>,
    SO: Data<Elem = A>,
    DS: Dimension + DimAdd<DO, Output = DSUM> + DimMax<Dim<[usize; 1]>, Output = DS>, // DS + DO = DSUM and DS >= 1
    DO: Dimension + DimAdd<DS, Output = DSUM> + DimMax<Dim<[usize; 1]>, Output = DO>, // DO + DS = DSUM and DO >= 1
    DR: Dimension,
    DSUM: Dimension + DimAbsSub<Dim<[usize; 2]>, Output = DR>, // DSUM - 2 = DR
{
    type Output = ArrayBase<OwnedRepr<A>, DR>;

    fn tensordot(&self, other: &ArrayBase<SO, DO>) -> Self::Output {
        ndarray_einsum_beta::tensordot(self, other, &[Axis(self.ndim() - 1)], &[Axis(0)]).into_dimensionality::<DR>().expect(
            "AutoDiff tensordot only supports contraction if the resulting dimension is small enough to be static")
    }
}

/// tensor contraction trait with arbitrary axes
/// using const generics to enforce that the number of axes is known at compile time
pub trait TensorContraction<const N: usize, T> {
    type Output;
    fn contract(&self, other: &T, axes: (&[usize; N], &[usize; N])) -> Self::Output;
}

macro_rules! impl_tensor_contraction {
    ($n:literal) => {
        impl<A, // element type, must be the same for both arrays and the result
             SS, // array storage for self
             DS, // self dimension
             SO, // other storage
             DO, // other dimension
             DR, // result dimension
             DSUBS, // DS minus the contraction axes
             DSUBO, // DO minus the contraction axes
            >
            TensorContraction<$n, ArrayBase<SO, DO>> for ArrayBase<SS, DS>
        where
            A: LinalgScalar,
            SS: Data<Elem = A>,
            SO: Data<Elem = A>,
            DS: Dimension
                + DimMax<Dim<[usize; $n]>, Output = DS> // DS >= N
                + DimAbsSub<Dim<[usize; $n]>, Output = DSUBS>, // DS - N = DSUBS
            DO: Dimension
                + DimMax<Dim<[usize; $n]>, Output = DO> // DO >= N
                + DimAbsSub<Dim<[usize; $n]>, Output = DSUBO>, // DO - N = DSUBO

            DSUBS: Dimension + DimAdd<DSUBO, Output = DR>, // DSUBS + DSUBO = DR
            DSUBO: Dimension + DimAdd<DSUBS, Output = DR>, // DSUBO + DSUBS = DR
            DR: Dimension,
        {
            type Output = ArrayBase<OwnedRepr<A>, DR>;

            fn contract(&self, other: &ArrayBase<SO, DO>, axes: (&[usize; $n], &[usize; $n])) -> Self::Output {
                let lhs_axes = axes.0.map(|i| Axis(i)).into_iter().collect::<Vec<_>>();
                let rhs_axes = axes.1.map(|i| Axis(i)).into_iter().collect::<Vec<_>>();

                ndarray_einsum_beta::tensordot(self, other, lhs_axes.as_slice(), rhs_axes.as_slice()).into_dimensionality::<DR>().expect(
                    "AutoDiff contract only supports contraction if the resulting dimension is small enough to be static")
            }
        }
    };
}

// ndarray only supports Dim<[usize; N]> for N <= 6, everything else is IxDyn
impl_tensor_contraction!(0);
impl_tensor_contraction!(1);
impl_tensor_contraction!(2);
impl_tensor_contraction!(3);
impl_tensor_contraction!(4);
impl_tensor_contraction!(5);
impl_tensor_contraction!(6);

// traits for sum, sum_axis, mean, mean_axis, var, var_axis, prod, and sort
pub trait Sum {
    type Output;
    fn sum(&self) -> Self::Output;
}

pub trait SumAxis {
    type Output;
    fn sum_axis(&self, axis: usize) -> Self::Output;
}

pub trait Mean {
    type Output;
    fn mean(&self) -> Self::Output;
}

pub trait MeanAxis {
    type Output;
    fn mean_axis(&self, axis: usize) -> Self::Output;
}

pub trait Var {
    type Output;
    fn var(&self) -> Self::Output;
}

pub trait VarAxis {
    type Output;
    fn var_axis(&self, axis: usize) -> Self::Output;
}

pub trait Prod {
    type Output;
    fn prod(&self) -> Self::Output;
}

pub trait Sort {
    type Output;
    fn sort(&self) -> Self::Output;
}

pub enum EighOrder {
    AlgebraicAscending,
    AlgebraicDescending,
    AbsoluteAscending,
    AbsoluteDescending,
}

pub trait Eigh {
    type Output;
    fn eigh(&self, uplo: ndarray_linalg::solveh::UPLO, order: EighOrder) -> Self::Output;
}

pub trait Eigvalsh {
    type Output;
    fn eigvalsh(&self, uplo: ndarray_linalg::solveh::UPLO, order: EighOrder) -> Self::Output;
}

/// Quadradic form is a function of the form f(x) = x^T A x
/// NOTE: x is real-valued
/// this trait should be implemented for the matrix A
/// so f(x) = A.quadradic_form(x)
pub trait QuadradicForm<T> {
    type Output;
    fn quadradic_form(&self, other: &T) -> Self::Output;
}

/// Hermitian quadradic form is a function of the form f(x) = x^H A x
/// NOTE: x is complex-valued
/// this trait should be implemented for the matrix A
/// so f(x) = A.hermitian_quadradic_form(x)
pub trait HermitianQuadradicForm<T> {
    type Output;
    fn hermitian_quadradic_form(&self, other: &T) -> Self::Output;
}

/// Bilinear form is a function of the form f(x, y) = x^T A y
/// NOTE: x and y are real-valued
/// this trait should be implemented for the matrix A
/// so f(x, y) = A.bilinear_form(x, y)
pub trait BilinearForm<T> {
    type Output;
    fn bilinear_form(&self, x: &T, y: &T) -> Self::Output;
}

/// Hermitian bilinear form is a function of the form f(x, y) = x^H A y
/// NOTE: x and y are complex-valued
/// this trait should be implemented for the matrix A
/// so f(x, y) = A.hermitian_bilinear_form(x, y)
pub trait HermitianBilinearForm<T> {
    type Output;
    fn hermitian_bilinear_form(&self, x: &T, y: &T) -> Self::Output;
}

#[test]
fn test_tensor_dot() {
    let a = ndarray::arr2(&[[1, 2], [3, 4]]);
    let b = ndarray::arr2(&[[5, 6], [7, 8]]);
    let c = a.tensordot(&b);
    assert_eq!(c, ndarray::arr2(&[[19, 22], [43, 50]]));
    assert_eq!(c, a.dot(&b));
}

#[test]
fn test_tensor_contraction() {
    let a = ndarray::arr2(&[[1, 2], [3, 4]]);
    let b = ndarray::arr2(&[[5, 6], [7, 8]]);
    let c = a.contract(&b, (&[1], &[0]));
    assert_eq!(c, ndarray::arr2(&[[19, 22], [43, 50]]));
    assert_eq!(c, a.dot(&b));
    let ct = a.contract(&b, (&[0], &[0]));
    assert_eq!(ct, a.t().dot(&b));

    let s = a.contract(&b, (&[0, 1], &[1, 0]));
    assert_eq!(s[()], a.dot(&b).diag().sum());

    let ex = a.contract(&b, (&[], &[]));

    let mut res = ndarray::Array4::zeros((2, 2, 2, 2));
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    res[[i, j, k, l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    assert_eq!(ex, res);
}
