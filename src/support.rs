//! Utility functions, support structures, etc.

use crate::bit_vector::BitVector;
use crate::int_vector::IntVector;
use crate::ops::{Vector, Access};
use crate::rl_vector::RLVector;
use crate::sparse_vector::SparseVector;
use crate::bits;

use std::ops::Range;

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

/// An index for narrowing down a range before binary search.
///
/// `SampleIndex` takes a strictly increasing sequence of `n` values in `0..universe`.
/// The first value must be `0`.
/// The index chooses a divisor that partitions the universe into approximately `n / 8` ranges.
/// Each range `i` contains the values in range `(i * divisor)..((i + 1) * divisor)`.
/// Given a value `x`, we can restrict the binary search to the range `self.range(x)` of values.
///
/// This is a simple support structure not intended to be serialized.
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::IntVector;
/// use simple_sds::ops::{Vector, Access};
/// use simple_sds::support::SampleIndex;
///
/// let source: Vec<usize> = vec![0, 33, 124, 131, 224, 291, 322, 341, 394, 466, 501];
/// let values = IntVector::from(source);
/// let index = SampleIndex::new(&values, 540);
///
/// let range = index.range(300);
/// assert!(values.get(range.start) <= 300);
/// assert!(values.get_or(range.end, 540) > 300);
/// ```
///
/// # Notes
///
/// * This is a simple support structure not intended to be serialized.
/// * While `SampleIndex` is intended to be used with [`IntVector`], the values are [`usize`] as they are usually interpreted as such in applications.
pub struct SampleIndex {
    divisor: usize,
    num_values: usize,
    samples: IntVector,
}

impl SampleIndex {
    /// Ratio of number of values to number of samples.
    pub const RATIO: usize = 8;

    /// Returns a `SampleIndex` for the given values.
    ///
    /// # Arguments
    ///
    /// * `values`: A strictly increasing sequence of values starting from `0`.
    /// * `universe`: Universe size. The values must be in range `0..universe`.
    ///
    /// # Panics
    ///
    /// Panics if the first value is not `0`, the sequence of values is not strictly increasing, or if universe size is too small for the values.
    pub fn new(values: &IntVector, universe: usize) -> Self {
        if values.is_empty() {
            return SampleIndex {
                divisor: usize::MAX,
                num_values: 0,
                samples: IntVector::with_len(1, 1, 0).unwrap(),
            }
        }

        // We must have a strictly increasing sequence of values and a large enough universe size.
        let mut prev = values.get(0);
        assert_eq!(prev, 0, "SampleIndex::new(): The initial value must be 0");
        for value in values.iter().skip(1) {
            assert!(prev < value, "SampleIndex::new(): The values must be strictly increasing");
            prev = value;
        }
        assert!(universe > (prev as usize), "SampleIndex::new(): Universe size ({}) must be larger than the last value ({})", universe, prev);

        let (num_samples, divisor) = Self::parameters(values.len(), universe);
        let width = bits::bit_len((values.len() - 1) as u64);
        let mut samples = IntVector::with_len(num_samples, width, 0).unwrap();

        // Invariant: `values[samples[i]] <= i * divisor` and `values[samples[i] + 1] > i * divisor`.
        let mut offset = 0;
        for sample in 1..samples.len() {
            let threshold = sample * divisor;
            while offset + 1 < values.len() && (values.get(offset + 1) as usize) <= threshold {
                offset += 1;
            }
            samples.set(sample, offset as u64);
        }

        SampleIndex {
            divisor,
            num_values: values.len(),
            samples,
        }
    }

    /// Returns a range that will contain the given value if it is present in the values.
    ///
    /// If `value < universe`, the range guarantees the following invariants:
    ///
    /// * `values.get(range.start) <= value`
    /// * `values.get_or(range.end, universe) > value`
    pub fn range(&self, value: usize) -> Range<usize> {
        let offset = value / self.divisor;
        let start = self.samples.get_or(offset, self.num_values as u64) as usize;
        // We need + 1 because the next sample may be too small.
        let limit = self.samples.get_or(offset + 1, self.num_values as u64) as usize + 1;
        start..limit
    }

    // Returns `(samples, divisor)` such that `i / divisor` for `i in 0..universe` maps evenly to `0..samples`.
    fn parameters(values: usize, universe: usize) -> (usize, usize) {
        let num_samples = bits::div_round_up(values, Self::RATIO);
        let divisor = bits::div_round_up(universe, num_samples);
        let num_samples = bits::div_round_up(universe, divisor);
        (num_samples, divisor)
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use rand::Rng;
    use rand::rngs::ThreadRng;

    fn check_parameters(values: usize, universe: usize) {
        let (samples, divisor) = SampleIndex::parameters(values, universe);
        assert!((samples - 1) * divisor < universe, "Divisor {} is too large with {} samples and universe size {}", divisor, samples, universe);
        assert!(samples * divisor >= universe, "Divisor {} is too small with {} samples and universe size {}", divisor, samples, universe);
        assert_eq!((universe - 1) / divisor, samples - 1, "Last value {} does not map to to last sample {} with {} values", universe - 1, samples - 1, values);
    }

    fn generate_values(universe: usize, density: usize, rng: &mut ThreadRng) -> IntVector {
        let mut values: Vec<usize> = Vec::new();
        let mut last = 0;
        while last < universe {
            values.push(last);
            last += 1 + rng.gen_range(0..density);
        }
        IntVector::from(values)
    }

    #[test]
    fn parameters() {
        let mut rng = rand::thread_rng();
        let mut values: Vec<usize> = (1..10).collect();
        for _ in 0..100 {
            values.push(rng.gen_range(1..10000));
        }

        for v in values {
            let mut universe: Vec<usize> = vec![v, v + 1, v * SampleIndex::RATIO - 1, v * SampleIndex::RATIO, v * SampleIndex::RATIO + 1];
            for _ in 0..10 {
                universe.push(v * rng.gen_range(1..10) + rng.gen_range(0..v));
            }
            for u in universe {
                check_parameters(v, u);
            }
        }
    }

    #[test]
    fn sample_index() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let universe = rng.gen_range(10_000..100_000);
            let density = rng.gen_range(2..1000);
            let values = generate_values(universe, density, &mut rng);
            let index = SampleIndex::new(&values, universe);
            for _ in 0..1000 {
                let query = rng.gen_range(0..universe);
                let range = index.range(query);
                let start = values.get(range.start) as usize;
                assert!(start <= query, "Range start values[{}] = {} is too large for query {} (universe size {}, density {})", range.start, start, query, universe, density);
                let end = values.get_or(range.end, universe as u64) as usize;
                assert!(end > query, "Range end values[{}] = {} is too small for query {} (universe size {}, density {})", range.end, end, query, universe, density);
            }
        }
    }
}

//-----------------------------------------------------------------------------
