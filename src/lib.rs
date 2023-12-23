pub mod adops;
pub mod arithmetic;
pub mod autodiff;
pub mod autodiffable;
pub mod func_traits;
pub mod funcs;
pub mod traits;
pub mod autotuple;

#[cfg(feature = "ndarray")]
pub mod ad_ndarray;

#[cfg(test)]
mod test_autodiff;
