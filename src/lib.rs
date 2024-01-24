#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod adops;
pub mod autodiff;
pub mod diffable;
pub mod compose;
pub mod autodiffable;
pub mod autotuple;
pub mod func_traits;
pub mod funcs;
pub mod traits;
pub mod forward;
pub mod gradienttype;

// re-export
pub use autodiff::*;
pub use diffable::*;
pub use compose::*;
pub use autodiffable::*;
pub use autotuple::*;
pub use func_traits::*;
//pub use funcs::*;
pub use traits::*;
pub use forward::*;
pub use gradienttype::*;

#[cfg(feature = "ndarray")]
pub mod ad_ndarray;

#[cfg(feature = "ndarray")]
pub use ad_ndarray::*;

#[cfg(test)]
mod test_autodiff;
#[cfg(test)]
mod test_manualdiff;

// re-export derive proc-macros
pub use autodiff_derive::*;
