use super::*;

use crate::raw_vector::{RawVector, AccessRaw, PushRaw};
use crate::serialize::Serialize;
use crate::{internal, serialize};

use std::cmp;

use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;

//-----------------------------------------------------------------------------

fn random_raw_vector(len: usize, density: f64) -> RawVector {
    let mut data = RawVector::with_capacity(len);
    let mut rng = rand::thread_rng();
    if density == 0.5 {
        while data.len() < len {
            unsafe { data.push_int(rng.gen(), cmp::min(len - data.len(), 64)); }
        }
    }
    else {
        let dist = Bernoulli::new(density).unwrap();
        let mut iter = dist.sample_iter(&mut rng);
        while data.len() < len {
            data.push_bit(iter.next().unwrap());
        }
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

//-----------------------------------------------------------------------------

fn try_iter(bv: &BitVector) {
    // Forward.
    internal::try_bitvec_iter(bv);

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

    // nth.
    for i in 0..bv.len() {
        assert_eq!(bv.iter().nth(i), Some(bv.get(i)), "Invalid nth({})", i);
    }

    // nth_back.
    for i in 0..bv.len() {
        assert_eq!(bv.iter().nth_back(i), Some(bv.get(bv.len() - 1 - i)), "Invalid nth_back({})", i);
    }
}

#[test]
fn empty_vector() {
    let empty = BitVector::from(RawVector::new());
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.count_ones(), 0, "Empty vector contains ones");
    assert!(empty.iter().next().is_none(), "Non-empty iterator from an empty vector");
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
fn from_iter() {
    let correct: Vec<bool> = vec![false, true, true, false, true, false, true, true, false, false, false];
    let bv: BitVector = correct.iter().cloned().collect();
    assert_eq!(bv.len(), correct.len(), "Invalid length for a bitvector built from an iterator");
    for i in 0..bv.len() {
        assert_eq!(bv.get(i), correct[i], "Invalid value {} in the bitvector", i);
    }
}

#[test]
fn copy_bit_vec() {
    let source = random_vector(1234, 0.35);
    let copy = BitVector::copy_bit_vec(&source);
    assert_eq!(copy, source, "Invalid copy created with copy_bit_vec()");
}

#[test]
fn iter() {
    let bv = random_vector(1563, 0.5);
    try_iter(&bv);
}

#[test]
fn serialize() {
    let bv = random_vector(2137, 0.5);
    let _ = serialize::test(&bv, "bitvector", Some(40), true);
}

#[test]
#[ignore]
fn large() {
    let bv = random_vector(9875321, 0.5);
    try_iter(&bv);
    let _ = serialize::test(&bv, "large-bitvector", Some(154308), true);
}

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
    let mut bv = random_vector(1957, 0.5);
    assert!(!bv.supports_rank(), "Rank support was enabled by default");
    bv.enable_rank();
    internal::try_rank(&bv);
}

#[test]
fn serialize_rank() {
    let mut bv = random_vector(1921, 0.5);
    bv.enable_rank();
    let _ = serialize::test(&bv, "bitvector-rank", Some(46), true);
}

#[test]
#[ignore]
fn large_rank() {
    let mut bv = random_vector(9871248, 0.5);
    bv.enable_rank();
    internal::try_rank(&bv);
    let _ = serialize::test(&bv, "large-bitvector-rank", Some(192806), true);
}

//-----------------------------------------------------------------------------

// Only test the non-standard functionality.
fn try_one_iter<T: Transformation>(bv: &BitVector) {
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

    // Skip forward by 1, 2, 4, 8, ... values.
    let bit_len = bits::bit_len(bv.count_ones() as u64);
    for k in 0..bit_len {
        let jump: usize = (1 << k) - 1;
        let mut iter = T::one_iter(bv);
        let mut skip_iter = T::one_iter(bv);
        while iter.len() > 0 {
            for _ in 0..jump {
                let _ = iter.next();
            }
            let iter_val = iter.next();
            let skip_val = skip_iter.nth(jump);
            assert_eq!(skip_val, iter_val, "Invalid value from OneIter::nth");
        }
    }
}

#[test]
fn empty_select() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_select(), "Select support was enabled by default");
    empty.enable_select();
    assert!(empty.supports_select(), "Failed to enable select support");
    assert!(empty.select_iter(empty.count_ones()).next().is_none(), "Got a result for select_iter past the end");
    assert!(empty.select_iter(empty.count_ones()).next().is_none(), "Got a result for select_iter past the end");
}

#[test]
fn nonempty_select() {
    let mut bv = random_vector(1957, 0.5);
    assert!(!bv.supports_select(), "Select support was enabled by default");
    bv.enable_select();
    assert!(bv.supports_select(), "Failed to enable select support");
    internal::try_select(&bv, 1);
}

#[test]
fn sparse_select() {
    let mut bv = random_vector(4200, 0.01);
    assert!(!bv.supports_select(), "Select support was enabled by default");
    bv.enable_select();
    internal::try_select(&bv, 1);
}

#[test]
fn one_iter() {
    let bv = random_vector(3122, 0.5);
    internal::try_one_iter(&bv, 1);
    try_one_iter::<Identity>(&bv);
}

#[test]
fn serialize_select() {
    let mut bv = random_vector(1921, 0.5);
    let old_size = bv.size_in_bytes();
    bv.enable_select();
    assert!(bv.size_in_bytes() > old_size, "Select support did not increase the size in bytes");
    let _ = serialize::test(&bv, "bitvector-select", None, true);
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

    internal::try_select(&bv, 1);
    internal::try_one_iter(&bv, 1);
    try_one_iter::<Identity>(&bv);
    let _ = serialize::test(&bv, "large-bitvector-select", None, true);
}

//-----------------------------------------------------------------------------

#[test]
fn empty_select_zero() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_select_zero(), "Select zero support was enabled by default");
    empty.enable_select_zero();
    assert!(empty.supports_select_zero(), "Failed to enable select zero support");
    assert!(empty.select_zero(Complement::count_ones(&empty)).is_none(), "Got a result for select past the end");
    assert!(empty.select_zero_iter(Complement::count_ones(&empty)).next().is_none(), "Got a result for select_iter past the end");
}

#[test]
fn nonempty_select_zero() {
    let mut bv = random_vector(2133, 0.5);
    assert!(!bv.supports_select_zero(), "Select zero support was enabled by default");
    bv.enable_select_zero();
    assert!(bv.supports_select_zero(), "Failed to enable select zero support");
    internal::try_select_zero(&bv);
}

#[test]
fn sparse_select_zero() {
    let mut bv = random_vector(3647, 0.99);
    assert!(!bv.supports_select_zero(), "Select zero support was enabled by default");
    bv.enable_select_zero();
    internal::try_select_zero(&bv);
}

#[test]
fn zero_iter() {
    let bv = random_vector(3354, 0.5);
    internal::try_zero_iter(&bv);
    try_one_iter::<Complement>(&bv);
}

#[test]
fn serialize_select_zero() {
    let mut bv = random_vector(1764, 0.5);
    let old_size = bv.size_in_bytes();
    bv.enable_select_zero();
    assert!(bv.size_in_bytes() > old_size, "Select zero support did not increase the size in bytes");
    let _ = serialize::test(&bv, "bitvector-select-zero", None, true);
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

    internal::try_select_zero(&bv);
    internal::try_zero_iter(&bv);
    try_one_iter::<Complement>(&bv);
    let _ = serialize::test(&bv, "large-bitvector-select-zero", None, true);
}

//-----------------------------------------------------------------------------

#[test]
fn empty_pred_succ() {
    let mut empty = BitVector::from(RawVector::new());
    assert!(!empty.supports_pred_succ(), "Predecessor/successor support was enabled by default");
    empty.enable_pred_succ();
    assert!(empty.supports_pred_succ(), "Failed to enable predecessor/successor support");
    assert!(empty.predecessor(0).next().is_none(), "Invalid predecessor at 0");
    assert!(empty.successor(empty.len()).next().is_none(), "Invalid successor at vector size");
}

#[test]
fn nonempty_pred_succ() {
    let mut bv = random_vector(2466, 0.5);
    assert!(!bv.supports_pred_succ(), "Predecessor/successor support was enabled by default");
    bv.enable_pred_succ();
    assert!(bv.supports_pred_succ(), "Failed to enable predecessor/successor support");
    internal::try_pred_succ(&bv);
}

#[test]
fn serialize_pred_succ() {
    let mut bv = random_vector(1893, 0.5);
    let old_size = bv.size_in_bytes();
    bv.enable_pred_succ();
    assert!(bv.size_in_bytes() > old_size, "Predecessor/successor support did not increase the size in bytes");
    let _ = serialize::test(&bv, "bitvector-pred-succ", None, true);
}

#[test]
#[ignore]
fn large_pred_succ() {
    let regions: Vec<(usize, f64)> = vec![(4096, 0.5), (4096, 0.01), (8192, 0.5), (8192, 0.01), (4096, 0.5)];
    let mut bv = non_uniform_vector(&regions, false);
    bv.enable_pred_succ();

    internal::try_pred_succ(&bv);
    let _ = serialize::test(&bv, "large-bitvector-pred-succ", None, true);
}

//-----------------------------------------------------------------------------
