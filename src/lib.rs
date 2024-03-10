#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod adops;
pub mod autodiff;
pub mod autodiffable;
pub mod autotuple;
pub mod compose;
pub mod diffable;
pub mod forward;
pub mod func_traits;
pub mod funcs;
pub mod gradienttype;
pub mod traits;

// re-export
pub use autodiff::*;
pub use autodiffable::*;
pub use autotuple::*;
pub use compose::*;
pub use diffable::*;
pub use func_traits::*;
//pub use funcs::*;
pub use forward::*;
pub use gradienttype::*;
pub use traits::*;

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
