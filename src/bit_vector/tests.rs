use super::*;

use crate::raw_vector::{RawVector, AccessRaw, PushRaw};
use crate::serialize::Serialize;
use crate::serialize;

use std::iter::{DoubleEndedIterator, ExactSizeIterator};
use std::fs;

use rand::distributions::{Bernoulli, Distribution};

//-----------------------------------------------------------------------------

fn random_raw_vector(len: usize, density: f64) -> RawVector {
    let mut data = RawVector::with_capacity(len);
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(density).unwrap();
    let mut iter = dist.sample_iter(&mut rng);
    while data.len() < len {
        data.push_bit(iter.next().unwrap());
    }
    assert_eq!(data.len(), len, "Invalid length for random RawVector");
    data
}

fn random_vector(len: usize, density: f64) -> BitVector {
    let data = random_raw_vector(len, density);
    let bv = BitVector::from(data);
    assert_eq!(bv.len(), len, "Invalid length for random BitVector");

    // This test assumes that the number of ones is within 6 stdevs of the expected.
    let ones: f64 = bv.count_ones() as f64;
    let expected: f64 = len as f64 * density;
    let stdev: f64 = (len as f64 * density * (1.0 - density)).sqrt();
    assert!(ones >= expected - 6.0 * stdev && ones <= expected + 6.0 * stdev,
        "random_vector({}, {}): unexpected number of ones: {}", len, density, ones);

    bv
}

// Each region is specified as (ones, density).
// The region continues with unset bits until we would generate the next set bit.
// The final bitvector may be complemented if necessary.
fn non_uniform_vector(regions: &[(usize, f64)], complement: bool) -> BitVector {
    let mut data = RawVector::new();
    let mut rng = rand::thread_rng();
    let mut total_ones: usize = 0;
    for (ones, density) in regions.iter() {
        let dist = Bernoulli::new(*density).unwrap();
        let mut generated: usize = 0;
        let mut iter = dist.sample_iter(&mut rng);
        loop {
            let bit = iter.next().unwrap();
            generated += bit as usize;
            if generated > *ones {
                break;
            }
            data.push_bit(bit);
        }
        total_ones += *ones;
    }

    if complement {
        data = data.complement();
        total_ones = data.len() - total_ones;
    }
    let bv = BitVector::from(data);
    assert_eq!(bv.count_ones(), total_ones, "Invalid number of ones in the non-uniform BitVector");

    bv
}

fn try_serialize(bv: &BitVector, base_name: &str, expected_size: Option<usize>) {
    if let Some(bytes) = expected_size {
        assert_eq!(bv.size_in_bytes(), bytes, "Invalid BitVector size in bytes");
    }

    let filename = serialize::temp_file_name(base_name);
    serialize::serialize_to(bv, &filename).unwrap();

    let copy: BitVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, *bv, "Serialization changed the BitVector");

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------

fn try_iter(bv: &BitVector) {
    assert_eq!(bv.iter().len(), bv.len(), "Invalid Iter length");

    // Forward.
    for (index, value) in bv.iter().enumerate() {
        assert_eq!(value, bv.get(index), "Invalid value {} (forward)", index);
    }

    // Backward.
    let mut index = bv.len();
    let mut iter = bv.iter();
    while let Some(value) = iter.next_back() {
        index -= 1;
        assert_eq!(value, bv.get(index), "Invalid value {} (backward)", index);
    }

    // Meet in the middle.
    let mut next = 0;
    let mut limit = bv.len();
    let mut iter = bv.iter();
    while iter.len() > 0 {
        assert_eq!(iter.next(), Some(bv.get(next)), "Invalid value {} (forward, bidirectional)", next);
        next += 1;
        if iter.len() == 0 {
            break;
        }
        limit -= 1;
        assert_eq!(iter.next_back(), Some(bv.get(limit)), "Invalid value {} (backward, bidirectional)", limit);
    }
    assert_eq!(next, limit, "Iterator did not visit all values");
}

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
fn access() {
    let data = random_raw_vector(1791, 0.5);
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

    let copy: Vec<bool> = bv.into_iter().collect();
    assert_eq!(copy, correct, "Iterator conversions changed the values");
}

#[test]
fn iter() {
    let bv = random_vector(1563, 0.5);
    try_iter(&bv);
}

#[test]
fn serialize() {
    let bv = random_vector(2137, 0.5);
    try_serialize(&bv, "bitvector", Some(320));
}

#[test]
#[ignore]
fn large() {
    let bv = random_vector(9875321, 0.5);
    try_iter(&bv);
    try_serialize(&bv, "large-bitvector", Some(1234464));
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

fn try_rank(bv: &BitVector) {
    assert!(bv.supports_rank(), "Failed to enable rank support");
    assert_eq!(bv.rank(bv.len()), bv.count_ones(), "Invalid rank at vector size");

    let mut rank: usize = 0;
    for i in 0..bv.len() {
        assert_eq!(bv.rank(i), rank, "Invalid rank at {}", i);
        rank += bv.get(i) as usize;
    }
}

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
    let mut bv = random_vector(1957, 0.5);
    assert!(!bv.supports_rank(), "Rank support was enabled by default");
    bv.enable_rank();
    try_rank(&bv);
}

#[test]
fn serialize_rank() {
    let mut bv = random_vector(1921, 0.5);
    bv.enable_rank();
    try_serialize(&bv, "bitvector-rank", Some(368));
}

#[test]
#[ignore]
fn large_rank() {
    let mut bv = random_vector(9871248, 0.5);
    bv.enable_rank();
    try_rank(&bv);
    try_serialize(&bv, "large-bitvector-rank", Some(1542448));
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

fn try_select(bv: &BitVector) {
    assert!(bv.supports_select(), "Failed to enable select support");
    assert_eq!(bv.select(bv.count_ones()).next(), None, "Got a result for select past the end");

    let mut next: usize = 0;
    for i in 0..bv.count_ones() {
        let value = bv.select(i).next().unwrap();
        assert_eq!(value.0, i, "Invalid rank for select({})", i);
        assert!(value.1 >= next, "select({}) == {}, expected at least {}", i, value.1, next);
        assert!(bv.get(value.1), "select({}) == {} is not set", i, value.1);
        next = value.1 + 1;
    }
}

fn try_one_iter<T: Transformation>(bv: &BitVector) {
    assert_eq!(T::one_iter(bv).len(), T::count_ones(bv), "Invalid OneIter length");

    // Iterate forward.
    let mut next: (usize, usize) = (0, 0);
    for (index, value) in T::one_iter(bv) {
        assert_eq!(index, next.0, "Invalid rank from OneIter (forward)");
        assert!(value >= next.1, "Too small value from OneIter (forward)");
        assert!(T::bit(bv, value), "OneIter returned an unset bit (forward)");
        next = (next.0 + 1, value + 1);
    }

    // Iterate backward.
    let mut limit: (usize, usize) = (T::count_ones(bv), bv.len());
    let mut iter = T::one_iter(bv);
    while let Some((index, value)) = iter.next_back() {
        assert_eq!(index, limit.0 - 1, "Invalid rank from OneIter (backward)");
        assert!(value < limit.1, "Too small value from OneIter (backward)");
        assert!(T::bit(bv, value), "OneIter returned an unset bit (backward)");
        limit = (limit.0 - 1, value);
    }

    // Meet in the middle.
    let mut next: (usize, usize) = (0, 0);
    let mut limit: (usize, usize) = (T::count_ones(bv), bv.len());
    let mut iter = T::one_iter(bv);
    while iter.len() > 0 {
        let (index, value) = iter.next().unwrap();
        assert_eq!(index, next.0, "Invalid rank from OneIter (forward, bidirectional)");
        assert!(value >= next.1, "Too small value from OneIter (forward, bidirectional)");
        assert!(T::bit(bv, value), "OneIter returned an unset bit (forward, bidirectional)");
        next = (next.0 + 1, value + 1);

        if iter.len() == 0 {
            break;
        }

        let (index, value) = iter.next_back().unwrap();
        assert_eq!(index, limit.0 - 1, "Invalid rank from OneIter (backward, bidirectional)");
        assert!(value < limit.1, "Too small value from OneIter (backward, bidirectional)");
        assert!(T::bit(bv, value), "OneIter returned an unset bit (backward, bidirectional)");
        limit = (limit.0 - 1, value);
    }
    assert_eq!(next.0, limit.0, "Iterator did not visit all values");
}

#[test]
fn empty_select() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_select(), "Select support was enabled by default");
    empty.enable_select();
    assert!(empty.supports_select(), "Failed to enable select support");
    assert_eq!(empty.select(empty.count_ones()).next(), None, "Got a result for select past the end");
}

#[test]
fn nonempty_select() {
    let mut bv = random_vector(1957, 0.5);
    assert!(!bv.supports_select(), "Select support was enabled by default");
    bv.enable_select();
    assert!(bv.supports_select(), "Failed to enable select support");
    try_select(&bv);
}

#[test]
fn sparse_select() {
    let mut bv = random_vector(4200, 0.01);
    assert!(!bv.supports_select(), "Select support was enabled by default");
    bv.enable_select();
    try_select(&bv);
}

#[test]
fn one_iter() {
    let bv = random_vector(3122, 0.5);
    try_one_iter::<Identity>(&bv);
}

#[test]
fn serialize_select() {
    let mut bv = random_vector(1921, 0.5);
    let old_size = bv.size_in_bytes();
    bv.enable_select();
    assert!(bv.size_in_bytes() > old_size, "Select support did not increase the size in bytes");
    try_serialize(&bv, "bitvector-select", None);
}

#[test]
#[ignore]
fn large_select() {
    let regions: Vec<(usize, f64)> = vec![(4096, 0.5), (4096, 0.01), (8192, 0.5), (8192, 0.01), (4096, 0.5)];
    let mut bv = non_uniform_vector(&regions, false);
    bv.enable_select();

    let ss = bv.select.as_ref().unwrap();
    assert_eq!(ss.superblocks(), 7, "Invalid number of select superblocks");
    assert_eq!(ss.long_superblocks(), 3, "Invalid number of long superblocks");
    assert_eq!(ss.short_superblocks(), 4, "Invalid number of short superblocks");

    try_select(&bv);
    try_one_iter::<Identity>(&bv);
    try_serialize(&bv, "large-bitvector-select", None);
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

fn try_select_zero(bv: &BitVector) {
    assert!(bv.supports_select_zero(), "Failed to enable select zero support");
    assert_eq!(bv.select_zero(Complement::count_ones(bv)).next(), None, "Got a result for select past the end");

    let mut next: usize = 0;
    for i in 0..Complement::count_ones(bv) {
        let value = bv.select_zero(i).next().unwrap();
        assert_eq!(value.0, i, "Invalid rank for select_zero({})", i);
        assert!(value.1 >= next, "select_zero({}) == {}, expected at least {}", i, value.1, next);
        assert!(!bv.get(value.1), "select_zero({}) == {} is set", i, value.1);
        next = value.1 + 1;
    }
}

#[test]
fn empty_select_zero() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_select_zero(), "Select zero support was enabled by default");
    empty.enable_select_zero();
    assert!(empty.supports_select_zero(), "Failed to enable select zero support");
    assert_eq!(empty.select_zero(Complement::count_ones(&empty)).next(), None, "Got a result for select past the end");
}

#[test]
fn nonempty_select_zero() {
    let mut bv = random_vector(2133, 0.5);
    assert!(!bv.supports_select_zero(), "Select zero support was enabled by default");
    bv.enable_select_zero();
    assert!(bv.supports_select_zero(), "Failed to enable select zero support");
    try_select_zero(&bv);
}

#[test]
fn sparse_select_zero() {
    let mut bv = random_vector(3647, 0.99);
    assert!(!bv.supports_select_zero(), "Select zero support was enabled by default");
    bv.enable_select_zero();
    try_select_zero(&bv);
}

#[test]
fn zero_iter() {
    let bv = random_vector(3354, 0.5);
    try_one_iter::<Complement>(&bv);
}

#[test]
fn serialize_select_zero() {
    let mut bv = random_vector(1764, 0.5);
    let old_size = bv.size_in_bytes();
    bv.enable_select_zero();
    assert!(bv.size_in_bytes() > old_size, "Select zero support did not increase the size in bytes");
    try_serialize(&bv, "bitvector-select-zero", None);
}

#[test]
#[ignore]
fn large_select_zero() {
    let regions: Vec<(usize, f64)> = vec![(4096, 0.5), (4096, 0.01), (8192, 0.5), (8192, 0.01), (4096, 0.5)];
    let mut bv = non_uniform_vector(&regions, true);
    bv.enable_select_zero();

    let ss = bv.select_zero.as_ref().unwrap();
    assert_eq!(ss.superblocks(), 7, "Invalid number of select superblocks");
    assert_eq!(ss.long_superblocks(), 3, "Invalid number of long superblocks");
    assert_eq!(ss.short_superblocks(), 4, "Invalid number of short superblocks");

    try_select_zero(&bv);
    try_one_iter::<Complement>(&bv);
    try_serialize(&bv, "large-bitvector-select-zero", None);
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------

fn try_pred_succ(bv: &BitVector) {
    assert!(bv.supports_pred_succ(), "Failed to enable predecessor/successor support");

    for i in 0..bv.len() {
        let rank = bv.rank(i);
        let pred_result = bv.predecessor(i).next();
        let succ_result = bv.successor(i).next();
        if bv.get(i) {
            assert_eq!(pred_result, Some((rank, i)), "Invalid predecessor result at a set bit");
            assert_eq!(succ_result, Some((rank, i)), "Invalid successor result at a set bit");
        } else {
            if rank == 0 {
                assert_eq!(pred_result, None, "Got a predecessor result before the first set bit");
            } else {
                if let Some((pred_rank, pred_value)) = pred_result {
                    let new_rank = bv.rank(pred_value);
                    assert_eq!(new_rank, rank - 1, "The returned value was not the predecessor");
                    assert_eq!(pred_rank, new_rank, "Predecessor returned an invalid rank");
                    assert!(bv.get(pred_value), "Predecessor returned an unset bit");
                } else {
                    panic!("Could not find a predecessor");
                }
            }
            if rank == bv.count_ones() {
                assert_eq!(succ_result, None, "Got a successor result after the last set bit");
            } else {
                if let Some((succ_rank, succ_value)) = succ_result {
                    let new_rank = bv.rank(succ_value);
                    assert_eq!(new_rank, rank, "The returned value was not the successor");
                    assert_eq!(succ_rank, new_rank, "Successor returned an invalid rank");
                    assert!(bv.get(succ_value), "Successor returned an unset bit");
                } else {
                    panic!("Could not find a successor");
                }
            }
        }
    }
}

#[test]
fn empty_pred_succ() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_pred_succ(), "Predecessor/successor support was enabled by default");
    empty.enable_pred_succ();
    assert!(empty.supports_pred_succ(), "Failed to enable predecessor/successor support");
    assert_eq!(empty.predecessor(0).next(), None, "Invalid predecessor at 0");
    assert_eq!(empty.successor(empty.len()).next(), None, "Invalid successor at vector size");
}

#[test]
fn nonempty_pred_succ() {
    let mut bv = random_vector(2466, 0.5);
    assert!(!bv.supports_pred_succ(), "Predecessor/successor support was enabled by default");
    bv.enable_pred_succ();
    assert!(bv.supports_pred_succ(), "Failed to enable predecessor/successor support");
    try_pred_succ(&bv);
}

#[test]
fn serialize_pred_succ() {
    let mut bv = random_vector(1893, 0.5);
    let old_size = bv.size_in_bytes();
    bv.enable_pred_succ();
    assert!(bv.size_in_bytes() > old_size, "Predecessor/successor support did not increase the size in bytes");
    try_serialize(&bv, "bitvector-pred-succ", None);
}

#[test]
#[ignore]
fn large_pred_succ() {
    let regions: Vec<(usize, f64)> = vec![(4096, 0.5), (4096, 0.01), (8192, 0.5), (8192, 0.01), (4096, 0.5)];
    let mut bv = non_uniform_vector(&regions, false);
    bv.enable_pred_succ();

    try_pred_succ(&bv);
    try_serialize(&bv, "large-bitvector-pred-succ", None);
}

// TODO benchmarks: repeated tests vs tests where the exact query depends on the previous result

//-----------------------------------------------------------------------------
