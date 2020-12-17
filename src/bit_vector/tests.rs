use super::*;

///use crate::ops::{BitVec, Rank, Select, Complement};
use crate::ops::{BitVec};
use crate::raw_vector::{RawVector, GetRaw, SetRaw, PushRaw};
use crate::serialize::Serialize;
use crate::bits;
use crate::serialize;

use std::iter::{DoubleEndedIterator, ExactSizeIterator};
use std::{cmp, fs};

use rand::Rng;

//-----------------------------------------------------------------------------

fn random_raw_vector(len: usize) -> RawVector {
    let mut data = RawVector::with_capacity(len);
    let mut rng = rand::thread_rng();
    while data.len() < len {
        let value: u64 =  rng.gen();
        let bits = cmp::min(bits::WORD_BITS, len - data.len());
        data.push_int(value, bits);
    }
    assert_eq!(data.len(), len, "Invalid length for random RawVector");
    data
}

fn random_vector(len: usize) -> BitVector {
    let data = random_raw_vector(len);
    let bv = BitVector::from(data);
    assert_eq!(bv.len(), len, "Invalid length for random BitVector");
    bv
}

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
fn access_bitvector() {
    let data = random_raw_vector(1791);
    let bv = BitVector::from(data.clone());
    assert_eq!(bv.len(), data.len(), "Invalid bitvector length");

    for i in 0..bv.len() {
        assert_eq!(bv.get(i), data.bit(i), "Invalid bit {}", i);
    }
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
    let bv = random_vector(1563);

    let mut index = bv.len();
    let mut iter = bv.iter();
    while let Some(value) = iter.next_back() {
        index -= 1;
        assert_eq!(value, bv.get(index), "Invalid value {} when iterating backwards", index);
    }

    let mut next = 0;
    let mut limit = bv.len();
    let mut iter = bv.iter();
    while iter.len() > 0 {
        assert_eq!(iter.next(), Some(bv.get(next)), "Invalid value {} (forward)", next);
        next += 1;
        if iter.len() == 0 {
            break;
        }
        limit -= 1;
        assert_eq!(iter.next_back(), Some(bv.get(limit)), "Invalid value {} (backward)", limit);
    }
    assert_eq!(next, limit, "Iterator did not visit all values");
}

#[test]
fn serialize_bitvector() {
    let original = random_vector(2137);
    assert_eq!(original.size_in_bytes(), 312, "Invalid BitVector size in bytes");

    let filename = serialize::temp_file_name("bitvector");
    serialize::serialize_to(&original, &filename).unwrap();

    let copy: BitVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the BitVector");

    fs::remove_file(&filename).unwrap();
}

#[test]
#[ignore]
fn large_bitvector() {
    let original = random_vector(9875321);
    for (index, value) in original.iter().enumerate() {
        assert_eq!(value, original.get(index), "Invalid value {} in the bitvector", index);
    }
    assert_eq!(original.size_in_bytes(), 1234456, "Invalid BitVector size in bytes");

    let filename = serialize::temp_file_name("large-bitvector");
    serialize::serialize_to(&original, &filename).unwrap();

    let copy: BitVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the BitVector");

    fs::remove_file(&filename).unwrap();
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

#[test]
fn empty_rank() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_rank(), "Rank support was enabled by default");
    empty.enable_rank();
    assert!(empty.supports_rank(), "Failed to enable rank support");
    assert_eq!(empty.rank(empty.len()), empty.count_ones(), "Invalid rank at vector size");
}

#[test]
fn nonempty_rank() {
    let mut bv = random_vector(1957);
    assert!(!bv.supports_rank(), "Rank support was enabled by default");
    bv.enable_rank();
    assert!(bv.supports_rank(), "Failed to enable rank support");
    assert_eq!(bv.rank(bv.len()), bv.count_ones(), "Invalid rank at vector size");

    let mut rank: usize = 0;
    for i in 0..bv.len() {
        assert_eq!(bv.rank(i), rank, "Invalid rank at {}", i);
        rank += bv.get(i) as usize;
    }
}

#[test]
fn serialize_rank() {
    let mut original = random_vector(1921);
    original.enable_rank();
    assert_eq!(original.size_in_bytes(), 360, "Invalid BitVector size with rank support");

    let filename = serialize::temp_file_name("bitvector-rank");
    serialize::serialize_to(&original, &filename).unwrap();

    let copy: BitVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the BitVector");

    fs::remove_file(&filename).unwrap();
}

#[test]
#[ignore]
fn large_rank() {
    let mut original = random_vector(9871248);
    original.enable_rank();
    assert_eq!(original.rank(original.len()), original.count_ones(), "Invalid rank at vector size");
    assert_eq!(original.size_in_bytes(), 1542440, "Invalid BitVector size in bytes");

    let mut rank: usize = 0;
    for i in 0..original.len() {
        assert_eq!(original.rank(i), rank, "Invalid rank at {}", i);
        rank += original.get(i) as usize;
    }

    let filename = serialize::temp_file_name("large-bitvector-rank");
    serialize::serialize_to(&original, &filename).unwrap();

    let copy: BitVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the BitVector");

    fs::remove_file(&filename).unwrap();
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

// FIXME tests: Select, OneIter, + Serialize
// FIXME large tests
// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

// FIXME tests: Complement, ZeroIter, + Serialize
// FIXME large tests
// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------
