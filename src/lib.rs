pub mod adops;
pub mod autodiff;
pub mod autodiffable;
pub mod func_traits;
pub mod funcs;
pub mod traits;
pub mod impls;
pub mod autotuple;

#[cfg(feature = "ndarray")]
pub mod ad_ndarray;

#[cfg(test)]
mod test_autodiff;
#[cfg(test)]
mod test_manualdiff;
