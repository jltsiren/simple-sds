//! An immutable run-length encoded integer vector supporting rank/select-type queries.

// FIXME document

use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, AccessIter, VectorIndex};
use crate::rl_vector::RLVector;
use crate::serialize::Serialize;
use crate::wavelet_matrix::wm_core::WMCore;
use crate::wavelet_matrix::wm_core;

use std::io::{Error, ErrorKind};
use std::iter::FusedIterator;
use std::io;

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

// FIXME: special run-based operations such as select_run for iterators
impl<'a> RLWM<'a> {
    // Returns the starting offset of the value after reordering.
    fn start(&self, value: <Self as Vector>::Item) -> usize {
        self.first.get(value as usize) as usize
    }

    // Computes `first` from an iterator.
    fn start_offsets<Iter: Iterator<Item = (u64, usize)>>(iter: Iter, len: usize, max_value: u64) -> IntVector {
        // Count the number of occurrences of each value.
        let mut counts: Vec<(u64, usize)> = Vec::with_capacity((max_value + 1) as usize);
        for i in 0..=max_value {
            counts.push((i, 0));
        }
        for (value, len) in iter {
            counts[value as usize].1 += len;
        }
        wm_core::counts_to_first(counts, len)
    }
}

impl<'a> From<Vec<(u64, usize)>> for RLWM<'a> {
    fn from(runs: Vec<(u64, usize)>) -> Self {
        let mut len = 0;
        let mut max_value = 0;
        for (value, length) in runs.iter() {
            len += *length;
            if *value > max_value {
                max_value = *value;
            }
        }
        let first = Self::start_offsets(runs.iter().copied(), len, max_value);
        let data = WMCore::from(runs);
        RLWM { len, data, first }
    }
}

//-----------------------------------------------------------------------------

impl<'a> Vector for RLWM<'a> {
    type Item = u64;

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn width(&self) -> usize {
        self.data.width()
    }

    #[inline]
    fn max_len(&self) -> usize {
        usize::MAX
    }
}

impl<'a> Access<'a> for RLWM<'a> {
    type Iter = AccessIter<'a, Self>;

    fn get(&self, index: usize) -> <Self as Vector>::Item {
        self.inverse_select(index).unwrap().1
    }

    fn iter(&'a self) -> Self::Iter {
        Self::Iter::new(self)
    }
}

impl<'a> VectorIndex<'a> for RLWM<'a> {
    type ValueIter = ValueIter<'a>;

    fn contains(&self, value: <Self as Vector>::Item) -> bool {
        (value as usize) < self.first.len() && self.start(value) < self.len()
    }

    fn rank(&self, index: usize, value: <Self as Vector>::Item) -> usize {
        if !self.contains(value) {
            return 0;
        }
        self.data.map_down_with(index, value) - self.start(value)
    }

    fn inverse_select(&self, index: usize) -> Option<(usize, <Self as Vector>::Item)> {
        self.data.map_down(index).map(|(index, value)| (index - self.start(value), value))
    }

    fn value_iter(&'a self, value: <Self as Vector>::Item) -> Self::ValueIter {
        Self::ValueIter {
            parent: self,
            value,
            rank: 0,
        }
    }

    fn value_of(iter: &Self::ValueIter) -> <Self as Vector>::Item {
        iter.value
    }

    fn select(&self, rank: usize, value: <Self as Vector>::Item) -> Option<usize> {
        if !self.contains(value) {
            return None;
        }
        self.data.map_up_with(self.start(value) + rank, value)
    }

    fn select_iter(&'a self, rank: usize, value: <Self as Vector>::Item) -> Self::ValueIter {
        Self::ValueIter {
            parent: self,
            value,
            rank,
        }
    }
}

impl<'a> Serialize for RLWM<'a> {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize(writer)?;
        self.first.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let data = WMCore::load(reader)?;
        if data.len() != len {
            return Err(Error::new(ErrorKind::InvalidData, "Core length does not match wavelet matrix length"));
        }

        let first = IntVector::load(reader)?;
        Ok(RLWM { len, data, first, })
    }

    fn size_in_elements(&self) -> usize {
        let mut result = self.len.size_in_elements();
        result += self.data.size_in_elements();
        result += self.first.size_in_elements();
        result
    }
}

//-----------------------------------------------------------------------------

// FIXME AccessIter to replace the default 

//-----------------------------------------------------------------------------

// FIXME Optimized ValueIter
#[derive(Clone, Debug)]
pub struct ValueIter<'a> {
    parent: &'a RLWM<'a>,
    value: <RLWM<'a> as Vector>::Item,
    // The first rank we have not seen.
    rank: usize,
}

impl<'a> Iterator for ValueIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rank >= self.parent.len() {
            return None;
        }
        if let Some(index) = self.parent.select(self.rank, self.value) {
            let result = Some((self.rank, index));
            self.rank += 1;
            result
        } else {
            self.rank = self.parent.len();
            None
        }
    }
}

impl<'a> FusedIterator for ValueIter<'a> {}

//-----------------------------------------------------------------------------

// FIXME RunIter?

//-----------------------------------------------------------------------------

// FIXME IntoIter

//-----------------------------------------------------------------------------