//! An index for speeding up rank/select queries in [`RLVector`].

use super::*;

use std::ops::Range;

//-----------------------------------------------------------------------------

/// An index for narrowing down a range before binary search.
///
/// `SampleIndex` takes a strictly increasing sequence of `n` values in `0..universe`.
/// The first value must be `0`.
/// The index chooses a divisor that partitions the universe into approximately `n / 8` ranges.
/// Each range `i` contains the values in range `(i * divisor)..((i + 1) * divisor)`.
/// Given a value `x`, we can restrict the binary search to the range `self.range(x)` of values.
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::IntVector;
/// use simple_sds::ops::{Vector, Access};
/// use simple_sds::rl_vector::index::SampleIndex;
///
/// let values: Vec<usize> = vec![0, 33, 124, 131, 224, 291, 322, 341, 394, 466, 501];
/// let index = SampleIndex::new(values.iter().copied(), 540);
///
/// let range = index.range(300);
/// assert!(values[range.start] <= 300);
/// let upper_bound = *values.get(range.end).unwrap_or(&540);
/// assert!(upper_bound > 300);
/// ```
///
/// # Notes
///
/// * This is a simple support structure not intended to be serialized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleIndex {
    num_values: usize,
    divisor: usize,
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
    pub fn new<T: Iterator<Item = usize> + ExactSizeIterator>(iter: T, universe: usize) -> Self {
        let mut iter = iter;
        let len = iter.len();
        if len == 0 || universe == 0 {
            return SampleIndex {
                num_values: 0,
                divisor: usize::MAX,
                samples: IntVector::with_len(1, 1, 0).unwrap(),
            }
        }

        let (num_samples, divisor) = Self::parameters(len, universe);
        let width = bits::bit_len((len - 1) as u64);
        let mut samples = IntVector::with_len(num_samples, width, 0).unwrap();

        // Invariant: `values[samples[i]] <= i * divisor` and `values[samples[i] + 1] > i * divisor`.
        let mut offset = 0;
        let mut prev = iter.next().unwrap();
        let mut next = iter.next();
        assert_eq!(prev, 0, "SampleIndex::new(): The initial value must be 0");
        for sample in 1..samples.len() {
            let threshold = sample * divisor;
            while next.is_some() {
                let value = next.unwrap();
                if value > threshold {
                    break;
                }
                assert!(prev < value, "SampleIndex::new(): The values must be strictly increasing");
                offset += 1;
                prev = value;
                next = iter.next();
            }
            samples.set(sample, offset as u64);
        }
        assert!(universe > prev, "SampleIndex::new(): Universe size ({}) must be larger than the last value ({})", universe, prev);

        SampleIndex {
            num_values: len,
            divisor,
            samples,
        }
    }

    /// Returns a range that will contain the given value if it is present in the values.
    ///
    /// If `value < universe`, the range guarantees the following invariants:
    ///
    /// * `values[range.start] <= value`
    /// * `values.get(range.end).unwrap_or(universe) > value`
    pub fn range(&self, value: usize) -> Range<usize> {
        let offset = value / self.divisor;
        let start = self.samples.get_or(offset, self.num_values as u64) as usize;
        let mut limit = self.samples.get_or(offset + 1, self.num_values as u64) as usize;
        // We need + 1 because the next sample may be too small for some values in the range.
        if limit < self.num_values {
            limit += 1;
        }
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

    fn generate_values(universe: usize, density: usize, rng: &mut ThreadRng) -> Vec<usize> {
        let mut values: Vec<usize> = Vec::new();
        let mut last = 0;
        while last < universe {
            values.push(last);
            last += 1 + rng.gen_range(0..density);
        }
        values
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
            let index = SampleIndex::new(values.iter().copied(), universe);
            for _ in 0..1000 {
                let query = rng.gen_range(0..universe);
                let range = index.range(query);
                let start = values[range.start];
                assert!(start <= query, "Range start values[{}] = {} is too large for query {} (universe size {}, density {})", range.start, start, query, universe, density);
                let end = if range.end >= values.len() { universe } else { values[range.end] };
                assert!(end > query, "Range end values[{}] = {} is too small for query {} (universe size {}, density {})", range.end, end, query, universe, density);
            }
        }
    }
}

//-----------------------------------------------------------------------------
