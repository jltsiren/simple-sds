//! Utility functions, support structures, etc.

use crate::bit_vector::BitVector;
use crate::rl_vector::RLVector;
use crate::sparse_vector::SparseVector;

//-----------------------------------------------------------------------------

// Conversions between bitvector types.
macro_rules! bitvector_conversion {
    ($source:ident, $target:ident) => {
        impl From<$source> for $target {
            fn from(source: $source) -> Self {
                $target::copy_bit_vec(&source)
            }
        }
    };
}

bitvector_conversion!(RLVector, BitVector);
bitvector_conversion!(SparseVector, BitVector);

bitvector_conversion!(BitVector, RLVector);
bitvector_conversion!(SparseVector, RLVector);

bitvector_conversion!(BitVector, SparseVector);
bitvector_conversion!(RLVector, SparseVector);

//-----------------------------------------------------------------------------
