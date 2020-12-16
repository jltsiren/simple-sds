use super::*;

///use crate::ops::{BitVec, Rank, Select, Complement};
use crate::ops::{BitVec};
use crate::raw_vector::{RawVector, GetRaw, SetRaw, PushRaw};
use crate::serialize::Serialize;
use crate::serialize;

use std::iter::{DoubleEndedIterator, ExactSizeIterator};
use std::fs;

//-----------------------------------------------------------------------------

#[test]
fn empty_vector() {
    let empty = BitVector::from(RawVector::new());
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.count_ones(), 0, "Empty vector contains ones");
    assert_eq!(empty.iter().next(), None, "Non-empty iterator from an empty vector");
}

#[test]
fn non_empty_vector() {
    let mut raw = RawVector::with_len(18, false);
    raw.set_bit(3, true); raw.set_bit(5, true); raw.set_bit(11, true); raw.set_bit(17, true);

    let bv = BitVector::from(raw.clone());
    assert!(!bv.is_empty(), "The bitvector is empty");
    assert_eq!(bv.len(), 18, "Invalid length for the bitvector");
    assert_eq!(bv.count_ones(), 4, "Invalid number of ones in the bitvector");
    assert_eq!(bv.iter().len(), bv.len(), "Invalid size hint from the iterator");
    for (index, value) in bv.iter().enumerate() {
        assert_eq!(value, raw.bit(index), "Invalid value {} in the bitvector", index);
    }

    let copy = RawVector::from(bv);
    assert_eq!(copy, raw, "BitVector changed the contents of the RawVector");
}

#[test]
fn iterator_conversions() {
    let correct: Vec<bool> = vec![false, true, true, false, true, false, true, true, false, false, false];
    let bv: BitVector = correct.iter().cloned().collect();
    assert_eq!(bv.len(), correct.len(), "Invalid length for a bitvector built from an iterator");

    for (index, value) in bv.iter().enumerate() {
        assert_eq!(value, correct[index], "Invalid value {} in the bitvector", index);
    }

    let copy: Vec<bool> = bv.into_iter().collect();
    assert_eq!(copy, correct, "Iterator conversions changed the values");
}

#[test]
fn double_ended_iterator() {
    let correct: Vec<bool> = vec![false, true, true, false, true, false, true, true, false, false, false];

    let bv: BitVector = correct.iter().cloned().collect();
    let mut index = correct.len();
    let mut iter = bv.iter();
    while let Some(value) = iter.next_back() {
        index -= 1;
        assert_eq!(value, correct[index], "Invalid value {} when iterating backwards", index);
    }

    let mut next = 0;
    let mut limit = correct.len();
    let mut iter = bv.iter();
    while iter.len() > 0 {
        assert_eq!(iter.next(), Some(correct[next]), "Invalid value {} (forward)", next);
        next += 1;
        if iter.len() == 0 {
            break;
        }
        limit -= 1;
        assert_eq!(iter.next_back(), Some(correct[limit]), "Invalid value {} (backward)", limit);
    }
    assert_eq!(next, limit, "Iterator did not visit all values");
}

#[test]
fn serialize_bit_vector() {
    let mut raw = RawVector::new();
    for i in 0..64 {
        raw.push_int(i * (i + 1) * (i + 2), 16);
    }

    let original = BitVector::from(raw);
    assert_eq!(original.size_in_bytes(), 160, "Invalid BitVector size in bytes");

    let filename = serialize::temp_file_name("bit-vector");
    serialize::serialize_to(&original, &filename).unwrap();

    let copy: BitVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the BitVector");

    fs::remove_file(&filename).unwrap();
}

// FIXME large tests
// FIXME benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

// FIXME tests: Rank + Serialize
// FIXME large tests
// FIXME benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

// FIXME tests: Select, OneIter + Serialize
// FIXME large tests
// FIXME benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

// FIXME tests: Complement, ZeroIter + Serialize
// FIXME large tests
// FIXME benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------
