pub mod adops;
pub mod autodiff;
pub mod autodiffable;
pub mod autotuple;
pub mod func_traits;
pub mod funcs;
pub mod traits;
pub mod forward;
pub mod gradienttype;

#[cfg(feature = "ndarray")]
pub mod ad_ndarray;

#[cfg(test)]
mod test_autodiff;
#[cfg(test)]
mod test_manualdiff;
