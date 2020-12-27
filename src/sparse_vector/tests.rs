use super::*;

use crate::bit_vector::BitVector;
use crate::raw_vector::{RawVector, PushRaw};
use crate::serialize;

use std::fs;

use rand::distributions::{Bernoulli, Distribution};

//-----------------------------------------------------------------------------

fn random_vector(ones: usize, density: f64) -> SparseVector {
    let mut data: Vec<usize> = Vec::new();
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(density).unwrap();
    let mut universe = 0;

    let mut iter = dist.sample_iter(&mut rng);
    loop {
        if iter.next().unwrap() {
            if data.len() >= ones {
                break;
            }
            data.push(universe);
        }
        universe += 1;
    }

    let mut builder = SparseBuilder::new(universe, data.len()).unwrap();
    builder.extend(data);

    SparseVector::try_from(builder).unwrap()
}

fn random_bit_vector(ones: usize, density: f64) -> BitVector {
    let mut raw: RawVector = RawVector::new();
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(density).unwrap();
    let mut generated = 0;

    let mut iter = dist.sample_iter(&mut rng);
    loop {
        let bit = iter.next().unwrap();
        generated += bit as usize;
        if generated > ones {
            break;
        }
        raw.push_bit(bit);
    }

    BitVector::from(raw)
}

fn zero_vector(len: usize) -> SparseVector {
    SparseVector::try_from(SparseBuilder::new(len, 0).unwrap()).unwrap()
}

fn one_vector(len: usize) -> SparseVector {
    let mut builder = SparseBuilder::new(len, len).unwrap();
    for i in 0..len {
        builder.set(i);
    }
    SparseVector::try_from(builder).unwrap()
}

//-----------------------------------------------------------------------------

fn try_iter(sv: &SparseVector) {
    assert_eq!(sv.iter().len(), sv.len(), "Invalid Iter length");

    // Forward.
    for (index, value) in sv.iter().enumerate() {
        assert_eq!(value, sv.get(index), "Invalid value {} (forward)", index);
    }

    // Backward.
    let mut index = sv.len();
    let mut iter = sv.iter();
    while let Some(value) = iter.next_back() {
        index -= 1;
        assert_eq!(value, sv.get(index), "Invalid value {} (backward)", index);
    }

    // Meet in the middle.
    let mut next = 0;
    let mut limit = sv.len();
    let mut iter = sv.iter();
    while iter.len() > 0 {
        assert_eq!(iter.next(), Some(sv.get(next)), "Invalid value {} (forward, bidirectional)", next);
        next += 1;
        if iter.len() == 0 {
            break;
        }
        limit -= 1;
        assert_eq!(iter.next_back(), Some(sv.get(limit)), "Invalid value {} (backward, bidirectional)", limit);
    }
    assert_eq!(next, limit, "Iterator did not visit all values");
}

fn try_serialize(sv: &SparseVector, base_name: &str) {
    let filename = serialize::temp_file_name(base_name);
    serialize::serialize_to(sv, &filename).unwrap();

    let copy: SparseVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, *sv, "Serialization changed the SparseVector");

    fs::remove_file(&filename).unwrap();
}

#[test]
fn empty_vector() {
    let empty = zero_vector(0);
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.count_ones(), 0, "Empty vector contains ones");
    assert_eq!(empty.iter().next(), None, "Non-empty iterator from an empty vector");
}

#[test]
fn non_empty_vector() {
    let mut raw = RawVector::with_len(18, false);
    raw.set_bit(3, true); raw.set_bit(5, true); raw.set_bit(11, true); raw.set_bit(17, true);
    let bv = BitVector::from(raw);

    let sv = SparseVector::copy_bit_vec(&bv);
    assert!(!sv.is_empty(), "The bitvector is empty");
    assert_eq!(sv.len(), 18, "Invalid length for the bitvector");
    assert_eq!(sv.count_ones(), 4, "Invalid number of ones in the bitvector");
    assert_eq!(sv.iter().len(), sv.len(), "Invalid size hint from the iterator");
    assert!(sv.iter().eq(bv.iter()), "Invalid values from the iterator");
}

#[test]
fn conversions() {
    let original = random_bit_vector(59, 0.015);

    let sv_copy = SparseVector::copy_bit_vec(&original);
    let sv_from = SparseVector::from(original.clone());
    assert_eq!(sv_copy, sv_from, "Different SparseVectors with different construction methods");

    let copy = BitVector::from(sv_copy);
    assert_eq!(copy, original, "SparseVector changed the contents of the BitVector");
}

#[test]
fn uniform_vector() {
    let zeros = zero_vector(1861);
    assert!(!zeros.is_empty(), "The zero vector is empty");
    assert_eq!(zeros.len(), 1861, "Invalid length for the zero vector");
    assert_eq!(zeros.count_ones(), 0, "Invalid number of ones in the zero vector");
    assert_eq!(zeros.iter().len(), zeros.len(), "Invalid size hint from the zero vector");
    assert_eq!(zeros.iter().filter(|b| !*b).count(), zeros.len(), "Some bits were set in the iterator");

    let ones = one_vector(2133);
    assert!(!ones.is_empty(), "The ones vector is empty");
    assert_eq!(ones.len(), 2133, "Invalid length for the ones vector");
    assert_eq!(ones.count_ones(), ones.len(), "Invalid number of ones in the ones vector");
    assert_eq!(ones.iter().len(), ones.len(), "Invalid size hint from the ones vector");
    assert_eq!(ones.iter().filter(|b| *b).count(), ones.len(), "Some bits were unset in the iterator");
}

#[test]
fn access() {
    let bv = random_bit_vector(67, 0.025);
    let sv = SparseVector::copy_bit_vec(&bv);
    assert_eq!(sv.len(), bv.len(), "Invalid bitvector length");

    for i in 0..sv.len() {
        assert_eq!(sv.get(i), bv.get(i), "Invalid bit {}", i);
    }
}

#[test]
fn iter() {
    let sv = random_vector(72, 0.02);
    try_iter(&sv);
}

#[test]
fn serialize() {
    let sv = random_vector(66, 0.01);
    try_serialize(&sv, "sparse-vector");
}

#[test]
#[ignore]
fn large() {
    let sv = random_vector(20179, 0.02);
    try_iter(&sv);
    try_serialize(&sv, "large-sparse-vector");
}

//-----------------------------------------------------------------------------

fn try_rank(sv: &SparseVector) {
    assert!(sv.supports_rank(), "Failed to enable rank support");
    assert_eq!(sv.rank(sv.len()), sv.count_ones(), "Invalid rank at vector size");

    let mut rank: usize = 0;
    for i in 0..sv.len() {
        assert_eq!(sv.rank(i), rank, "Invalid rank at {}", i);
        rank += sv.get(i) as usize;
    }
}

#[test]
fn empty_rank() {
    let empty = zero_vector(0);
    assert_eq!(empty.rank(empty.len()), empty.count_ones(), "Invalid rank at vector size");
}

#[test]
fn nonempty_rank() {
    let sv = random_vector(81, 0.025);
    try_rank(&sv);
}

#[test]
fn uniform_rank() {
    let zeros = zero_vector(1977);
    let ones = one_vector(1654);
    try_rank(&zeros);
    try_rank(&ones);
}

#[test]
#[ignore]
fn large_rank() {
    let sv = random_vector(19666, 0.015);
    try_rank(&sv);
}

//-----------------------------------------------------------------------------

fn try_select(sv: &SparseVector) {
    assert!(sv.supports_select(), "Failed to enable select support");
    assert_eq!(sv.select(sv.count_ones()), None, "Got a result for select past the end");
    assert_eq!(sv.select_iter(sv.count_ones()).next(), None, "Got a result for select_iter past the end");

    let mut next: usize = 0;
    for i in 0..sv.count_ones() {
        let value = sv.select_iter(i).next().unwrap();
        assert_eq!(value.0, i, "Invalid rank for select_iter({})", i);
        assert!(value.1 >= next, "select_iter({}) == {}, expected at least {}", i, value.1, next);
        let index = sv.select(i).unwrap();
        assert_eq!(index, value.1, "Different results for select({}) and select_iter({})", i, i);
        assert!(sv.get(index), "Bit select({}) == {} is not set", i, index);
        next = value.1 + 1;
    }
}

fn try_one_iter(sv: &SparseVector) {
    assert_eq!(sv.one_iter().len(), sv.count_ones(), "Invalid OneIter length");

    // Iterate forward.
    let mut next: (usize, usize) = (0, 0);
    for (index, value) in sv.one_iter() {
        assert_eq!(index, next.0, "Invalid rank from OneIter (forward)");
        assert!(value >= next.1, "Too small value from OneIter (forward)");
        assert!(sv.get(value), "OneIter returned an unset bit (forward)");
        next = (next.0 + 1, value + 1);
    }

    // Iterate backward.
    let mut limit: (usize, usize) = (sv.count_ones(), sv.len());
    let mut iter = sv.one_iter();
    while let Some((index, value)) = iter.next_back() {
        assert_eq!(index, limit.0 - 1, "Invalid rank from OneIter (backward)");
        assert!(value < limit.1, "Too small value from OneIter (backward)");
        assert!(sv.get(value), "OneIter returned an unset bit (backward)");
        limit = (limit.0 - 1, value);
    }

    // Meet in the middle.
    let mut next: (usize, usize) = (0, 0);
    let mut limit: (usize, usize) = (sv.count_ones(), sv.len());
    let mut iter = sv.one_iter();
    while iter.len() > 0 {
        let (index, value) = iter.next().unwrap();
        assert_eq!(index, next.0, "Invalid rank from OneIter (forward, bidirectional)");
        assert!(value >= next.1, "Too small value from OneIter (forward, bidirectional)");
        assert!(sv.get(value), "OneIter returned an unset bit (forward, bidirectional)");
        next = (next.0 + 1, value + 1);

        if iter.len() == 0 {
            break;
        }

        let (index, value) = iter.next_back().unwrap();
        assert_eq!(index, limit.0 - 1, "Invalid rank from OneIter (backward, bidirectional)");
        assert!(value < limit.1, "Too small value from OneIter (backward, bidirectional)");
        assert!(sv.get(value), "OneIter returned an unset bit (backward, bidirectional)");
        limit = (limit.0 - 1, value);
    }
    assert_eq!(next.0, limit.0, "Iterator did not visit all values");
}

#[test]
fn empty_select() {
    let empty = zero_vector(0);
    assert_eq!(empty.select(empty.count_ones()), None, "Got a result for select past the end");
    assert_eq!(empty.select_iter(empty.count_ones()).next(), None, "Got a result for select_iter past the end");
}

#[test]
fn nonempty_select() {
    let sv = random_vector(70, 0.02);
    try_select(&sv);
}

#[test]
fn uniform_select() {
    let zeros = zero_vector(2020);
    let ones = one_vector(1984);
    try_select(&zeros);
    try_select(&ones);
}

#[test]
fn one_iter() {
    let sv = random_vector(102, 0.03);
    try_one_iter(&sv);
}

#[test]
#[ignore]
fn large_select() {
    let sv = random_vector(20304, 0.02);
    try_select(&sv);
    try_one_iter(&sv);
}

//-----------------------------------------------------------------------------

fn try_pred_succ(sv: &SparseVector) {
    assert!(sv.supports_pred_succ(), "Failed to enable predecessor/successor support");

    for i in 0..sv.len() {
        let rank = sv.rank(i);
        let pred_result = sv.predecessor(i).next();
        let succ_result = sv.successor(i).next();
        if sv.get(i) {
            assert_eq!(pred_result, Some((rank, i)), "Invalid predecessor result at a set bit");
            assert_eq!(succ_result, Some((rank, i)), "Invalid successor result at a set bit");
        } else {
            if rank == 0 {
                assert_eq!(pred_result, None, "Got a predecessor result before the first set bit");
            } else {
                if let Some((pred_rank, pred_value)) = pred_result {
                    let new_rank = sv.rank(pred_value);
                    assert_eq!(new_rank, rank - 1, "The returned value was not the predecessor");
                    assert_eq!(pred_rank, new_rank, "Predecessor returned an invalid rank");
                    assert!(sv.get(pred_value), "Predecessor returned an unset bit");
                } else {
                    panic!("Could not find a predecessor");
                }
            }
            if rank == sv.count_ones() {
                assert_eq!(succ_result, None, "Got a successor result after the last set bit");
            } else {
                if let Some((succ_rank, succ_value)) = succ_result {
                    let new_rank = sv.rank(succ_value);
                    assert_eq!(new_rank, rank, "The returned value was not the successor");
                    assert_eq!(succ_rank, new_rank, "Successor returned an invalid rank");
                    assert!(sv.get(succ_value), "Successor returned an unset bit");
                } else {
                    panic!("Could not find a successor");
                }
            }
        }
    }
}

#[test]
fn empty_pred_succ() {
    let empty = zero_vector(0);
    assert_eq!(empty.predecessor(0).next(), None, "Invalid predecessor at 0");
    assert_eq!(empty.successor(empty.len()).next(), None, "Invalid successor at vector size");
}

#[test]
fn nonempty_pred_succ() {
    let sv = random_vector(91, 0.025);
    try_pred_succ(&sv);
}

#[test]
fn uniform_pred_succ() {
    let zeros = zero_vector(1999);
    let ones = one_vector(2021);
    try_pred_succ(&zeros);
    try_pred_succ(&ones);
}

#[test]
#[ignore]
fn large_pred_succ() {
    let sv = random_vector(15663, 0.015);
    try_pred_succ(&sv);
}

//-----------------------------------------------------------------------------
