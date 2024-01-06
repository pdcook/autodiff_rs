use num::complex::Complex;

/// Compile-time calculation for what the gradient type should be based
/// on input and output types.
///
/// For most cases, the gradient type is the same as the output type.
/// However, for multi-parameter functions, like functions of arrays, this
/// is not the case
///
/// `<InputType as GradientType<OutputType>>::GradientType`
pub trait GradientType<OutputType> {
    /// The type of the gradient for a function with input type `Self` and output type `OutputType`
    type GradientType;
}

// macro to implement GradientType for a simple type
// where the gradient type is the same as the output type
macro_rules! impl_simple_types {
    ($($type:ty),*) => {
        $(
            // impl for values
            impl<T> GradientType<T> for $type {
                type GradientType = T;
            }
        )*
    };
}

// implement GradientType for all primitive types as well as complex numbers
impl_simple_types!(
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    isize,
    Complex<f32>,
    Complex<f64>
);
