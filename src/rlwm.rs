//! An immutable run-length encoded integer vector supporting rank/select-type queries.

// FIXME document

use crate::int_vector::IntVector;
use crate::rl_vector::RLVector;
use crate::wavelet_matrix::wm_core::WMCore;

// FIXME tests

//-----------------------------------------------------------------------------

// FIXME document, example
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RLWM<'a> {
    len: usize,
    data: WMCore<'a, RLVector>,
    // Starting offset of each value after reordering by the wavelet matrix, or `len` if the value does not exist.
    first: IntVector,
}

// FIXME construction from runs of (value, length)

// FIXME special run-based operations?

//-----------------------------------------------------------------------------

// FIXME Vector, Access, VectorIndex, Serialize

//-----------------------------------------------------------------------------

// FIXME AccessIter to replace the default 

//-----------------------------------------------------------------------------

// FIXME ValueIter

//-----------------------------------------------------------------------------

// FIXME RunIter

//-----------------------------------------------------------------------------

// FIXME ValueIter

//-----------------------------------------------------------------------------
