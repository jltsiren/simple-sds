//! The core wavelet matrix structure.

// FIXME document

use crate::bit_vector::BitVector;
use crate::ops::{BitVec, Rank, Select, SelectZero, PredSucc};
use crate::raw_vector::{RawVector, PushRaw};
use crate::serialize::Serialize;
use crate::bits;

use std::io::{Read, Write};
use std::{cmp, io};

//-----------------------------------------------------------------------------

// FIXME document, example
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WMCore {
    levels: Vec<BitVector>,
}

// FIXME impl
impl WMCore {
    /// Returns the length of each level in the structure.
    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    /// Returns the number of levels in the structure.
    pub fn width(&self) -> usize {
        self.levels.len()
    }

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

    /// Maps up with the given value.
    ///
    /// Returns the position in the original vector, or [`None`] if there is no such position.
    /// This is the position where [`Self::map_down`] would return `(index, value)`.
    ///
    /// # Arguments
    ///
    /// * `index`: Position in the reordered vector.
    /// * `value`: Value of an item.
    pub fn map_up_with(&self, index: usize, value: u64) -> Option<usize> {
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
    fn map_up_one(&self, index: usize, level: usize) -> Option<usize> {
        self.levels[level].select(index - self.levels[level].count_zeros())
    }

    // Maps the index from the next level with an unset bit.
    fn map_up_zero(&self, index: usize, level: usize) -> Option<usize> {
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

macro_rules! wm_core_from {
    ($t:ident) => {
        impl From<Vec<$t>> for WMCore {
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

                let mut result = WMCore { levels, };
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

//-----------------------------------------------------------------------------

impl Serialize for WMCore {
    fn serialize_header<T: Write>(&self, _: &mut T) -> io::Result<()> {
        Ok(())
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        let width = self.width();
        width.serialize(writer)?;
        for bv in self.levels.iter() {
            bv.serialize(writer)?;
        }
        Ok(())
    }

    // FIXME sanity check: each len is same
    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let width = usize::load(reader)?;
        let mut levels: Vec<BitVector> = Vec::with_capacity(width);
        for _ in 0..width {
            let bv = BitVector::load(reader)?;
            levels.push(bv);
        }
        let mut result = WMCore { levels, };
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

// FIXME tests

//-----------------------------------------------------------------------------
