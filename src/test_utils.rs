// Utility functions for tests.

use crate::bits;

use rand::Rng;

//-----------------------------------------------------------------------------

// Returns a vector of `len` random `width`-bit integers.
pub fn random_vector(len: usize, width: usize) -> Vec<u64> {
    let mut result: Vec<u64> = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..len {
        let value: u64 = rng.gen();
        result.push(value & bits::low_set(width));
    }
    result
}

//-----------------------------------------------------------------------------
