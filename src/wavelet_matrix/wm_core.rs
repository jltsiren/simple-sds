//! The core wavelet matrix structure.
//!
//! A wavelet matrix core is a vector of [`u64`] items.
//! It represents a bidirectional mapping between the items in the original vector and the same items in the reordered vector.
//! The reordered vector is based on sorting the items stably by their reverse binary representations.
//! Mapping down means mapping from the original vector to the reordered vector, while mapping up means the reverse.
//!
//! This structure can be understood as the positional BWT (PBWT) of the binary sequences corresponding to the integers.
//! Each level in the core wavelet matrix corresponds to a column in the PBWT.
//! Bitvector `bv[level]` on level `level` represent bit values
//!
//! > `1 << (width - 1 - level)`.
//!
//! If `bv[level][i] == 0`, position `i` on level `level` it maps to position
//!
//! > `bv[level].rank_zero(i)`
//!
//! on level `level + 1`.
//! Otherwise it maps to position
//!
//! > `bv[level].count_zeros() + bv[level].rank(i)`.

use crate::bit_vector::BitVector;
use crate::int_vector::IntVector;
use crate::ops::{BitVec, FullBitVec, Pack};
use crate::raw_vector::{RawVector, PushRaw};
use crate::rl_vector::{RLVector, RLBuilder};
use crate::serialize::Serialize;
use crate::bits;

use std::io::{Error, ErrorKind};
use std::{cmp, io};

//-----------------------------------------------------------------------------

/// A bidirectional mapping between the original vector and a stably sorted vector of the same items.
///
/// Each item consists of the lowest 1 to 64 bits of a [`u64`] value, as specified by the width of the vector.
/// The width determines the number of levels in the `WMCore`.
///
/// A `WMCore` can be built using the [`From`]:
///
/// * From a [`Vec`] of unsigned integers when the underlying bitvector type is [`BitVector`].
/// * From a [`Vec`] of `(item, length)` runs of [`u64`] integers when the underlying bitvector type is [`RLVector`].
///
/// The construction requires several passes over the input and uses the input vector as working space.
/// Using smaller integer types helps reducing the space overhead during construction.
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
/// use simple_sds::wavelet_matrix::wm_core::WMCore;
///
/// // Source data
/// let source: Vec<u64> = vec![1, 0, 3, 1, 1, 2, 4, 5, 1, 2, 1, 7, 0, 1];
/// let mut reordered: Vec<u64> = source.clone();
/// reordered.sort_by_key(|x| x.reverse_bits());
///
/// // Construction
/// let core: WMCore<'_, BitVector> = WMCore::from(source.clone());
/// assert_eq!(core.len(), source.len());
/// assert_eq!(core.width(), 3);
///
/// // Map down
/// for i in 0..core.len() {
///     let mapped = core.map_down_with(i, source[i]);
///     assert_eq!(reordered[mapped], source[i]);
///     assert_eq!(core.map_down(i), Some((mapped, source[i])));
/// }
///
/// // Map up
/// for i in 0..core.len() {
///     let mapped = core.map_up_with(i, reordered[i]).unwrap();
///     assert_eq!(source[mapped], reordered[i]);
///     assert_eq!(core.map_down_with(mapped, source[mapped]), i);
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WMCore<'a, T: FullBitVec<'a>> {
    levels: Vec<T>,
    _phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T: FullBitVec<'a>> WMCore<'a, T> {
    /// Returns the length of each level in the structure.
    #[inline]
    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    /// Returns `true` if the structure is empty.
    ///
    /// Keeps Clippy happy.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of levels in the structure.
    #[inline]
    pub fn width(&self) -> usize {
        self.levels.len()
    }

    // TODO: this should use inverse_select when it's implemented
    /// Maps the item at the given position down.
    ///
    /// Returns the position of the item in the reordered vector and the value of the item, or [`None`] if the position is invalid.
    pub fn map_down(&self, index: usize) -> Option<(usize, u64)> {
        if index >= self.len() {
            return None;
        }

        let mut index = index;
        let mut value = 0;
        for level in 0..self.width() {
            if self.levels[level].get(index) {
                index = self.map_down_one(index, level);
                value += self.bit_value(level);
            } else {
                index = self.map_down_zero(index, level);
            }
        }

        Some((index, value))
    }

    /// Maps down with the given value.
    ///
    /// Returns the position in the reordered vector.
    /// This is the number of occurrences of `value` before `index` in the original vector, plus the number of items that map before `value` in the reordering.
    ///
    /// # Arguments
    ///
    /// * `index`: Position in the original vector.
    /// * `value`: Value of an item.
    pub fn map_down_with(&self, index: usize, value: u64) -> usize {
        let mut index = cmp::min(index, self.len());
        for level in 0..self.width() {
            if value & self.bit_value(level) != 0 {
                index = self.map_down_one(index, level);
            } else {
                index = self.map_down_zero(index, level);
            }
        }

        index
    }

    /// Maps two positions down at once.
    ///
    /// Returns the positions in the reordered vector.
    /// Because [`Self::map_down_with`] is constrained by memory latency, this can be faster than doing two independent queries.
    ///
    /// # Arguments
    ///
    /// * `first`: A position in the original vector.
    /// * `second`: A position in the original vector.
    /// * `value`: Value of an item.
    pub fn map_down_with_two_positions(&self, first: usize, second: usize, value: u64) -> (usize, usize) {
        let mut first = cmp::min(first, self.len());
        let mut second = cmp::min(second, self.len());
        for level in 0..self.width() {
            if value & self.bit_value(level) != 0 {
                first = self.map_down_one(first, level);
                second = self.map_down_one(second, level);
            } else {
                first = self.map_down_zero(first, level);
                second = self.map_down_zero(second, level);
            }
        }

        (first, second)
    }

    /// Maps up with the given value.
    ///
    /// Returns the position in the original vector, or [`None`] if there is no such position.
    /// This is the position where [`Self::map_down`] would return `(index, value)`.
    ///
    /// # Arguments
    ///
    /// * `index`: Position in the reordered vector.
    /// * `value`: Value of an item.
    pub fn map_up_with(&'a self, index: usize, value: u64) -> Option<usize> {
        let mut index = index;
        for level in (0..self.width()).rev() {
            if value & self.bit_value(level) != 0 {
                index = self.map_up_one(index, level)?;
            } else {
                index = self.map_up_zero(index, level)?;
            }
        }

        Some(index)
    }

    // Returns the bit value for the given level.
    fn bit_value(&self, level: usize) -> u64 {
        1 << (self.width() - 1 - level)
    }

    // Maps the index to the next level with a set bit.
    fn map_down_one(&self, index: usize, level: usize) -> usize {
        self.levels[level].count_zeros() + self.levels[level].rank(index)
    }

    // Maps the index to the next level with an unset bit.
    fn map_down_zero(&self, index: usize, level: usize) -> usize {
        self.levels[level].rank_zero(index)
    }

    // Maps the index from the next level with a set bit.
    fn map_up_one(&'a self, index: usize, level: usize) -> Option<usize> {
        self.levels[level].select(index - self.levels[level].count_zeros())
    }

    // Maps the index from the next level with an unset bit.
    fn map_up_zero(&'a self, index: usize, level: usize) -> Option<usize> {
        self.levels[level].select_zero(index)
    }

    // Initializes the support structures for the bitvectors.
    fn init_support(&mut self) {
        for bv in self.levels.iter_mut() {
            bv.enable_rank();
            bv.enable_select();
            bv.enable_select_zero();
            bv.enable_pred_succ();
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> WMCore<'a, RLVector> {
    // Maps the index to the next level, using the bit value in the given position.
    // Returns (new position, bit value, remaining run length).
    // Returns `None` if the index is out of bounds.
    fn map_down_level_with_run(&self, index: usize, level: usize) -> Option<(usize, bool, usize)> {
        let (mut index, bit_value, run_len) = self.levels[level].rank_with_run(index)?;
        if bit_value {
            index += self.levels[level].count_zeros();
        }
        Some((index, bit_value, run_len))
    }

    /// Maps the item at the given position down.
    ///
    /// The return value consists of:
    ///
    /// * The position of the item in the reordered vector.
    /// * The value of the item.
    /// * The length of the right-maximal run containing the item.
    ///
    /// Returns [`None`] if the position is out of bounds.
    pub fn map_down_with_run(&self, index: usize) -> Option<(usize, u64, usize)> {
        if index >= self.len() {
            return None;
        }

        let mut index = index;
        let mut value = 0;
        let mut run_len = usize::MAX;
        for level in 0..self.width() {
            let (new_index, bit_value, level_run_len) = self.map_down_level_with_run(index, level)?;
            index = new_index;
            if bit_value {
                value += self.bit_value(level);
            }
            run_len = cmp::min(run_len, level_run_len);
        }

        Some((index, value, run_len))
    }


    // Maps the index from the next level with a set bit.
    // Returns (new position, remaining run length).
    // Returns `None` if the index is out of bounds.
    fn map_up_one_with_run(&'a self, index: usize, level: usize) -> Option<(usize, usize)> {
        self.levels[level].select_run(index - self.levels[level].count_zeros())
    }

    // Maps the index from the next level with an unset bit.
    // Returns (new position, remaining run length).
    // Returns `None` if the index is out of bounds.
    fn map_up_zero_with_run(&'a self, index: usize, level: usize) -> Option<(usize, usize)> {
        self.levels[level].select_zero_run(index)
    }

    /// Maps up with the given value.
    ///
    /// Returns the position in the original vector, as well as the length of the right-maximal run containing the item.
    /// Returns [`None`] if there is no such position.
    /// This is the position where [`Self::map_down_with_run`] would return `(index, value, run_len)`.
    ///
    /// # Arguments
    ///
    /// * `index`: Position in the reordered vector.
    /// * `value`: Value of an item.
    pub fn map_up_with_run(&'a self, index: usize, value: u64) -> Option<(usize, usize)> {
        let mut index = index;
        let mut run_len = usize::MAX;
        for level in (0..self.width()).rev() {
            if value & self.bit_value(level) != 0 {
                let (new_index, level_run_len) = self.map_up_one_with_run(index, level)?;
                index = new_index;
                run_len = cmp::min(run_len, level_run_len);
            } else {
                let (new_index, level_run_len) = self.map_up_zero_with_run(index, level)?;
                index = new_index;
                run_len = cmp::min(run_len, level_run_len);
            }
        }

        Some((index, run_len))
    }
}

//-----------------------------------------------------------------------------

// TODO: Optimize construction. There are fast algorithms in the literature.
macro_rules! wm_core_from {
    ($t:ident) => {
        impl From<Vec<$t>> for WMCore<'_, BitVector> {
            fn from(source: Vec<$t>) -> Self {
                let mut source = source;
                let max_value = source.iter().cloned().max().unwrap_or(0);
                let width = bits::bit_len(max_value as u64);

                let mut levels: Vec<BitVector> = Vec::new();
                for level in 0..width {
                    let bit_value: $t = 1 << (width - 1 - level);
                    let mut zeros: Vec<$t> = Vec::new();
                    let mut ones: Vec<$t> = Vec::new();
                    let mut raw_data = RawVector::with_capacity(source.len());

                    // Determine if the current bit is set in each value.
                    for value in source.iter() {
                        if value & bit_value != 0 {
                            ones.push(*value);
                            raw_data.push_bit(true);
                        } else {
                            zeros.push(*value);
                            raw_data.push_bit(false);
                        }
                    }

                    // Sort the values stably by the current bit.
                    source.clear();
                    source.extend(zeros);
                    source.extend(ones);
        
                    // Create the bitvector for the current level.
                    levels.push(BitVector::from(raw_data));
                }

                let mut result = WMCore { levels, _phantom: std::marker::PhantomData, };
                result.init_support();
                result
            }
        }
    }
}

wm_core_from!(u8);
wm_core_from!(u16);
wm_core_from!(u32);
wm_core_from!(u64);
wm_core_from!(usize);

impl From<Vec<(u64, usize)>> for WMCore<'_, RLVector> {
    fn from(runs: Vec<(u64, usize)>) -> Self {
        let len: usize = runs.iter().map(|(_, len)| *len).sum();
        let max_value = runs.iter().map(|(value, _)| *value).max().unwrap_or(0);
        let width = bits::bit_len(max_value);

        let mut runs = runs;
        let mut levels: Vec<RLVector> = Vec::new();
        for level in 0..width {
            let bit_value = 1 << (width - 1 - level);
            let mut zeros: Vec<(u64, usize)> = Vec::new();
            let mut ones: Vec<(u64, usize)> = Vec::new();
            let mut builder = RLBuilder::new();

            // Determine if the current bit is set in each run.
            let mut offset = 0;
            for (value, len) in runs.iter() {
                if value & bit_value != 0 {
                    unsafe { builder.set_run_unchecked(offset, *len); }
                    ones.push((*value, *len));
                } else {
                    zeros.push((*value, *len));
                }
                offset += len;
            }

            // Sort the values stably by the current bit.
            runs.clear();
            runs.extend(zeros);
            runs.extend(ones);

            // Create the bitvector for the current level.
            builder.set_len(len);
            levels.push(RLVector::from(builder));
        }

        let mut result = WMCore { levels, _phantom: std::marker::PhantomData, };
        result.init_support();
        result
    }
}

//-----------------------------------------------------------------------------

impl<'a, T: FullBitVec<'a>> Serialize for WMCore<'a, T> {
    fn serialize_header<W: io::Write>(&self, _: &mut W) -> io::Result<()> {
        Ok(())
    }

    fn serialize_body<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let width = self.width();
        width.serialize(writer)?;
        for bv in self.levels.iter() {
            bv.serialize(writer)?;
        }
        Ok(())
    }

    fn load<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let width = usize::load(reader)?;
        if width == 0 || width > bits::WORD_BITS {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid width"));
        }

        let mut len: Option<usize> = None;
        let mut levels: Vec<T> = Vec::with_capacity(width);
        for _ in 0..width {
            let bv = T::load(reader)?;
            match len {
                Some(len) => {
                    if bv.len() != len {
                        return Err(Error::new(ErrorKind::InvalidData, "Invalid bitvector length"));
                    }
                },
                None => {
                    len = Some(bv.len());
                },
            }
            levels.push(bv);
        }
        let mut result = WMCore { levels, _phantom: std::marker::PhantomData, };
        result.init_support();
        Ok(result)
    }

    fn size_in_elements(&self) -> usize {
        let mut result = 1; // Width.
        for bv in self.levels.iter() {
            result += bv.size_in_elements();
        }
        result
    }
}

//-----------------------------------------------------------------------------

// TODO: tests?
// TODO: take BTreeMap as input, handle missing items internally?
// Transforms the occurrence counts of items into starting positions in the reordered vector.
//
// The input consists of pairs `counts[i] = (i, count)`.
// For items with no occurrences, the starting position is the end of the vector (`len`).
pub(crate) fn counts_to_first(counts: Vec<(u64, usize)>, len: usize) -> IntVector {
    // Sort the counts in reverse bit order to get the order below the final level.
    let mut counts = counts;
    counts.sort_unstable_by_key(|(value, _)| value.reverse_bits());

    // Replace occurrence counts with the prefix sum in the sorted order.
    let mut cumulative = 0;
    for (_, count) in counts.iter_mut() {
        if *count == 0 {
            *count = len;
        } else {
            let increment = *count;
            *count = cumulative;
            cumulative += increment;
        }
    }

    // Sort the prefix sums by symbol to get starting offsets and then return the offsets.
    counts.sort_unstable_by_key(|(value, _)| *value);
    let mut result: IntVector = counts.into_iter().map(|(_, offset)| offset).collect();
    result.pack();
    result
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{internal, serialize};

    fn reordered_vector(original: &[u64]) -> Vec<u64> {
        let mut result = Vec::from(original);
        result.sort_unstable_by_key(|x| x.reverse_bits());
        result
    }

    fn check_core<'a, T: FullBitVec<'a>>(core: &'a WMCore<'a, T>, original: &[u64]) {
        assert_eq!(core.len(), original.len(), "Invalid WMCore length");
        let reordered = reordered_vector(original);

        for i in 0..core.len() {
            let mapped = core.map_down_with(i, original[i]);
            assert_eq!(reordered[mapped], original[i], "Invalid value at the mapped down position from {}", i);
            assert_eq!(core.map_down(i), Some((mapped, original[i])), "The map down functions are inconsistent at {}", i);
        }

        for i in 0..core.len() {
            let mapped = core.map_up_with(i, reordered[i]).unwrap();
            assert_eq!(original[mapped], reordered[i], "Invalid value at the mapped up position from {}", i);
            assert_eq!(core.map_down_with(mapped, original[mapped]), i, "Did not get the original position after mapping up and down from {}", i);
        }

        let first = internal::random_queries(core.len(), core.len());
        let second = internal::random_queries(core.len(), core.len());
        let values = internal::random_vector(core.len(), core.width());
        for i in 0..core.len() {
            let mapped = core.map_down_with_two_positions(first[i], second[i], values[i]);
            assert_eq!(mapped.0, core.map_down_with(first[i], values[i]), "Invalid first mapped position at {}, value {}", first[i], values[i]);
            assert_eq!(mapped.1, core.map_down_with(second[i], values[i]), "Invalid second mapped position at {}, value {}", second[i], values[i]);
        }
    }

    fn check_core_rl(core: &WMCore<'_, RLVector>, original: &[(u64, usize)]) {
        let mut values: Vec<u64> = Vec::new();
        for (value, len) in original.iter() {
            for _ in 0..*len {
                values.push(*value);
            }
        }
        check_core(core, &values);

        // map_down_with_run
        for i in 0..core.len() {
            let result = core.map_down_with_run(i);
            assert!(result.is_some(), "map_down_with_run returned None at {}", i);
            let (mapped, value, run_len) = result.unwrap();
            let (true_mapped, true_value) = core.map_down(i).unwrap();
            assert_eq!(mapped, true_mapped, "map_down_with_run returned a wrong mapped position at {}", i);
            assert_eq!(value, true_value, "map_down_with_run returned a wrong value at {}", i);
            for j in 0..run_len {
                assert_eq!(values.get(i + j), Some(&value), "map_down_with_run returned a too long run at {}, offset {}", i, j);
            }
            assert_ne!(values.get(i + run_len), Some(&value), "map_down_with_run returned a non-maximal run at {}", i);
        }
        let result = core.map_down_with_run(core.len());
        assert!(result.is_none(), "map_down_with_run returned a result for index past the end (len {})", core.len());

        // map_up_with_run
        let reordered = reordered_vector(&values);
        for i in 0..core.len() {
            let result = core.map_up_with_run(i, reordered[i]);
            assert!(result.is_some(), "map_up_with_run returned None at {}", i);
            let (mapped, run_len) = result.unwrap();
            assert_eq!(values[mapped], reordered[i], "map_up_with_run returned a wrong mapped position at {}", i);
            let (down_index, _, down_run_len) = core.map_down_with_run(mapped).unwrap();
            assert_eq!(down_index, i, "map_up_with_run returned a position that did not map back correctly at {}", i);
            assert_eq!(run_len, down_run_len, "map_up_with_run returned a wrong run length at {}", i);
            if i + 1 < core.len() && reordered[i + 1] != reordered[i] {
                let result = core.map_up_with_run(i + 1, reordered[i]);
                assert!(result.is_none(), "map_up_with_run returned a past-the-end result for value {}", reordered[i]);
            }
        }
        let result = core.map_up_with_run(core.len(), 0);
        assert!(result.is_none(), "map_up_with_run returned a result for index past the end (len {})", core.len());
    }

    #[test]
    fn empty_core_plain() {
        let original: Vec<u64> = Vec::new();
        let core: WMCore<'_, BitVector> = WMCore::from(original.clone());
        check_core(&core, &original);
    }

    #[test]
    fn empty_core_rl() {
        let original: Vec<(u64, usize)> = Vec::new();
        let core: WMCore<'_, RLVector> = WMCore::from(original.clone());
        check_core_rl(&core, &original);
    }

    #[test]
    fn non_empty_core_plain() {
        let original = internal::random_vector(322, 7);
        let core: WMCore<'_, BitVector> = WMCore::from(original.clone());
        check_core(&core, &original);
    }

    #[test]
    fn non_empty_core_rl() {
        let original = internal::random_integer_runs(37, 9, 0.08);
        let core: WMCore<'_, RLVector> = WMCore::from(original.clone());
        check_core_rl(&core, &original);
    }

    #[test]
    fn serialize_core_plain() {
        let original = internal::random_vector(286, 6);
        let core: WMCore<'_, BitVector> = WMCore::from(original.clone());
        serialize::test(&core, "wm-core-plain", None, true);
    }

    #[test]
    fn serialize_core_rl() {
        let original = internal::random_integer_runs(29, 8, 0.07);
        let core: WMCore<'_, RLVector> = WMCore::from(original.clone());
        serialize::test(&core, "wm-core-rl", None, true);
    }
}

//-----------------------------------------------------------------------------
