#[rustfmt::skip]
pub mod adops; // rustfmt breaks this module

#[rustfmt::skip]
pub mod autodiff; // rustfmt breaks this module

pub mod autodiffable;
pub mod autotuple;
pub mod func_traits;
pub mod funcs;
pub mod impls;
pub mod traits;

#[cfg(feature = "ndarray")]
pub mod ad_ndarray;

#[cfg(test)]
mod test_autodiff;
#[cfg(test)]
mod test_manualdiff;
