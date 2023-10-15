use super::*;

use crate::{internal, serialize};

//-----------------------------------------------------------------------------

fn random_vector(ones: usize, density: f64) -> SparseVector {
    let (positions, universe) = internal::random_positions(ones, density);

    let mut builder = SparseBuilder::new(universe, positions.len()).unwrap();
    assert!(!builder.is_multiset(), "Builder created with new() is a multiset");
    builder.extend(positions);

    SparseVector::try_from(builder).unwrap()
}

fn random_bit_vector(ones: usize, density: f64) -> BitVector {
    let (positions, universe) = internal::random_positions(ones, density);

    let mut raw: RawVector = RawVector::with_len(universe, false);
    for position in positions.iter() {
        raw.set_bit(*position, true);
    }

    BitVector::from(raw)
}

fn zero_vector(len: usize) -> SparseVector {
    let builder = SparseBuilder::new(len, 0).unwrap();
    assert!(!builder.is_multiset(), "Builder created with new() is a multiset");
    SparseVector::try_from(builder).unwrap()
}

fn one_vector(len: usize) -> SparseVector {
    let mut builder = SparseBuilder::new(len, len).unwrap();
    assert!(!builder.is_multiset(), "Builder created with new() is a multiset");
    for i in 0..len {
        builder.set(i);
    }
    SparseVector::try_from(builder).unwrap()
}

//-----------------------------------------------------------------------------

fn try_iter(sv: &SparseVector) {
    // Forward.
    internal::try_bitvec_iter(sv);

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

#[test]
fn empty_vector() {
    let empty = zero_vector(0);
    assert!(!empty.is_multiset(), "Empty vector is a multiset");
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.count_ones(), 0, "Empty vector contains ones");
    assert_eq!(empty.count_zeros(), 0, "Empty vector contains zeros");
    assert!(empty.iter().next().is_none(), "Non-empty iterator from an empty vector");
}

#[test]
fn non_empty_vector() {
    let mut raw = RawVector::with_len(18, false);
    raw.set_bit(3, true); raw.set_bit(5, true); raw.set_bit(11, true); raw.set_bit(17, true);
    let bv = BitVector::from(raw);

    let sv = SparseVector::copy_bit_vec(&bv);
    assert!(!sv.is_multiset(), "The bitvector is a multiset");
    assert!(!sv.is_empty(), "The bitvector is empty");
    assert_eq!(sv.len(), 18, "Invalid length for the bitvector");
    assert_eq!(sv.count_ones(), 4, "Invalid number of ones in the bitvector");
    assert_eq!(sv.count_zeros(), 14, "Invalid number of zeros in the bitvector");
    assert_eq!(sv.iter().len(), sv.len(), "Invalid size hint from the iterator");
    assert!(sv.iter().eq(bv.iter()), "Invalid values from the iterator");
}

#[test]
fn conversions() {
    let original = random_vector(59, 0.015);
    let bv = BitVector::copy_bit_vec(&original);
    let copy = SparseVector::copy_bit_vec(&bv);
    assert_eq!(copy, original, "Conversions changed the contents of the SparseVector");
}

#[test]
fn uniform_vector() {
    let zeros = zero_vector(1861);
    assert!(!zeros.is_multiset(), "The zero vector is a multiset");
    assert!(!zeros.is_empty(), "The zero vector is empty");
    assert_eq!(zeros.len(), 1861, "Invalid length for the zero vector");
    assert_eq!(zeros.count_ones(), 0, "Invalid number of ones in the zero vector");
    assert_eq!(zeros.count_zeros(), zeros.len(), "Invalid number of zeros in the zero vector");
    assert_eq!(zeros.iter().len(), zeros.len(), "Invalid size hint from the zero vector");
    assert_eq!(zeros.iter().filter(|b| !*b).count(), zeros.len(), "Some bits were set in the iterator");

    let ones = one_vector(2133);
    assert!(!ones.is_multiset(), "The one vector is a multiset");
    assert!(!ones.is_empty(), "The ones vector is empty");
    assert_eq!(ones.len(), 2133, "Invalid length for the ones vector");
    assert_eq!(ones.count_ones(), ones.len(), "Invalid number of ones in the ones vector");
    assert_eq!(ones.count_zeros(), 0, "Invalid number of zeros in the ones vector");
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
    let _ = serialize::test(&sv, "sparse-vector", None, true);
}

#[test]
#[ignore]
fn large() {
    let sv = random_vector(20179, 0.02);
    try_iter(&sv);
    let _ = serialize::test(&sv, "large-sparse-vector", None, true);
}

//-----------------------------------------------------------------------------

#[test]
fn empty_rank() {
    let empty = zero_vector(0);
    assert_eq!(empty.rank(empty.len()), empty.count_ones(), "Invalid rank at vector size");
}

#[test]
fn nonempty_rank() {
    let sv = random_vector(81, 0.025);
    internal::try_rank(&sv);
}

#[test]
fn uniform_rank() {
    let zeros = zero_vector(1977);
    let ones = one_vector(1654);
    internal::try_rank(&zeros);
    internal::try_rank(&ones);
}

#[test]
#[ignore]
fn large_rank() {
    let sv = random_vector(19666, 0.015);
    internal::try_rank(&sv);
}

//-----------------------------------------------------------------------------

fn try_one_iter(sv: &SparseVector, increment: usize) {
    // Iterate forward.
    internal::try_one_iter(sv, increment);

    // Iterate backward.
    let mut limit: (usize, usize) = (sv.count_ones(), sv.len());
    let mut iter = sv.one_iter();
    while let Some((index, value)) = iter.next_back() {
        assert_eq!(index, limit.0 - 1, "Invalid rank from OneIter (backward)");
        assert!(value < limit.1, "Too small value from OneIter (backward)");
        assert!(sv.get(value), "OneIter returned an unset bit (backward)");
        limit = (limit.0 - 1, value + (1 - increment));
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
        next = (next.0 + 1, value + increment);

        if iter.len() == 0 {
            break;
        }

        let (index, value) = iter.next_back().unwrap();
        assert_eq!(index, limit.0 - 1, "Invalid rank from OneIter (backward, bidirectional)");
        assert!(value < limit.1, "Too small value from OneIter (backward, bidirectional)");
        assert!(sv.get(value), "OneIter returned an unset bit (backward, bidirectional)");
        limit = (limit.0 - 1, value + (1 - increment));
    }
    assert_eq!(next.0, limit.0, "Iterator did not visit all values");
}

#[test]
fn empty_select() {
    let empty = zero_vector(0);
    assert!(empty.select(empty.count_ones()).is_none(), "Got a result for select past the end");
    assert!(empty.select_iter(empty.count_ones()).next().is_none(), "Got a result for select_iter past the end");
}

#[test]
fn nonempty_select() {
    let sv = random_vector(70, 0.02);
    internal::try_select(&sv, 1);
}

#[test]
fn uniform_select() {
    let zeros = zero_vector(2020);
    let ones = one_vector(1984);
    internal::try_select(&zeros, 1);
    internal::try_select(&ones, 1);
}

#[test]
fn one_iter() {
    let sv = random_vector(102, 0.03);
    try_one_iter(&sv, 1);
}

#[test]
#[ignore]
fn large_select() {
    let sv = random_vector(20304, 0.02);
    internal::try_select(&sv, 1);
    try_one_iter(&sv, 1);
}

//-----------------------------------------------------------------------------

#[test]
fn empty_select_zero() {
    let empty = zero_vector(0);
    assert!(empty.select_zero(empty.count_zeros()).is_none(), "Got a result for select_zero past the end");
    assert!(empty.select_zero_iter(empty.count_zeros()).next().is_none(), "Got a result for select_zero_iter past the end");
}

#[test]
fn nonempty_select_zero() {
    let sv = random_vector(77, 0.025);
    internal::try_select_zero(&sv);
}

#[test]
fn uniform_select_zero() {
    let zeros = zero_vector(1998);
    let ones = one_vector(2022);
    internal::try_select_zero(&zeros);
    internal::try_select_zero(&ones);
}

#[test]
fn zero_iter() {
    let sv = random_vector(97, 0.02);
    internal::try_zero_iter(&sv);
}

#[test]
#[ignore]
fn large_select_zero() {
    let sv = random_vector(19664, 0.022);
    internal::try_select_zero(&sv);
    internal::try_zero_iter(&sv);
}

//-----------------------------------------------------------------------------

#[test]
fn empty_pred_succ() {
    let empty = zero_vector(0);
    assert!(empty.predecessor(0).next().is_none(), "Invalid predecessor at 0");
    assert!(empty.successor(empty.len()).next().is_none(), "Invalid successor at vector size");
}

#[test]
fn nonempty_pred_succ() {
    let sv = random_vector(91, 0.025);
    internal::try_pred_succ(&sv);
}

#[test]
fn uniform_pred_succ() {
    let zeros = zero_vector(1999);
    let ones = one_vector(2021);
    internal::try_pred_succ(&zeros);
    internal::try_pred_succ(&ones);
}

#[test]
#[ignore]
fn large_pred_succ() {
    let sv = random_vector(15663, 0.015);
    internal::try_pred_succ(&sv);
}

//-----------------------------------------------------------------------------

fn multiset_access(sv: &SparseVector, truth: &[usize]) {
    let mut offset: usize = 0;
    for i in 0..sv.len() {
        while offset < truth.len() && truth[offset] < i {
            offset += 1;
        }
        let expected = offset < truth.len() && truth[offset] == i;
        assert_eq!(sv.get(i), expected, "Invalid bit at {}", i);
    }
}

fn multiset_rank(sv: &SparseVector, truth: &[usize]) {
    assert!(sv.supports_rank(), "Failed to enable rank support");
    assert_eq!(sv.rank(sv.len()), sv.count_ones(), "Invalid rank at vector size");

    let mut rank: usize = 0;
    let mut offset: usize = 0;
    for i in 0..sv.len() {
        assert_eq!(sv.rank(i), rank, "Invalid rank at {}", i);
        while offset < truth.len() && truth[offset] == i {
            rank += 1; offset += 1;
        }
    }
}

fn multiset_pred_succ(sv: &SparseVector, truth: &[usize]) {
    assert!(sv.supports_pred_succ(), "Failed to enable predecessor/successor support");

    let mut rank: usize = 0;
    let mut offset: usize = 0;
    for i in 0..sv.len() {
        let mut count: usize = 0;
        while offset < truth.len() && truth[offset] == i {
            count += 1; offset += 1;
        }
        let pred_result = sv.predecessor(i).next();
        let succ_result = sv.successor(i).next();
        if count > 0 {
            assert_eq!(pred_result, Some((rank + count - 1, i)), "Invalid predecessor result at a set bit");
            assert_eq!(succ_result, Some((rank, i)), "Invalid successor result at a set bit");
        } else {
            if rank == 0 {
                assert!(pred_result.is_none(), "Got a predecessor result before the first set bit");
            } else {
                if let Some((pred_rank, pred_value)) = pred_result {
                    assert_eq!(pred_rank, rank - 1, "Predecessor returned an invalid rank");
                    assert!(sv.get(pred_value), "Predecessor returned an unset bit");
                } else {
                    panic!("Could not find a predecessor");
                }
            }
            if rank == sv.count_ones() {
                assert!(succ_result.is_none(), "Got a successor result after the last set bit");
            } else {
                if let Some((succ_rank, succ_value)) = succ_result {
                    assert_eq!(succ_rank, rank, "Successor returned an invalid rank");
                    assert!(sv.get(succ_value), "Successor returned an unset bit");
                } else {
                    panic!("Could not find a successor");
                }
            }
        }
        rank += count;
    }

    if sv.len() > 0 {
        assert_eq!(sv.predecessor(sv.len()).next(), sv.predecessor(sv.len() - 1).next(), "Invalid predecessor at vector size");
    }
    assert!(sv.successor(sv.len()).next().is_none(), "Invalid successor at vector size");
}

fn multiset_tests(sv: &SparseVector, len: usize, truth: &[usize]) {
    assert!(sv.is_multiset(), "The bitvector is not a multiset");
    assert!(!sv.is_empty(), "The bitvector is empty");
    assert_eq!(sv.len(), len, "Invalid length for the bitvector");
    assert_eq!(sv.count_ones(), truth.len(), "Invalid number of ones in the bitvector");

    multiset_access(sv, truth);
    try_iter(sv);
    let _ = serialize::test(sv, "multiset-sparse-vector", None, true);

    multiset_rank(sv, &truth);
    internal::try_select(sv, 0);
    try_one_iter(sv, 0);
    multiset_pred_succ(sv, &truth);
}

#[test]
fn builder_multiset() {
    let source: Vec<usize> = vec![123, 131, 131, 131, 347, 961];
    let mut builder = SparseBuilder::multiset(1024, source.len());
    assert!(builder.is_multiset(), "Builder created with multiset() is not a multiset");
    builder.extend(source.iter().cloned());
    assert!(builder.is_full(), "Full builder is not full");
    let sv = SparseVector::try_from(builder).unwrap();
    multiset_tests(&sv, 1024, &source);
}

#[test]
fn iter_multiset() {
    let source: Vec<usize> = vec![115, 432, 432, 641, 951, 951];
    let sv = SparseVector::try_from_iter(source.iter().cloned()).unwrap();
    multiset_tests(&sv, 952, &source);
}

#[test]
fn overfull_multiset() {
    let source: Vec<usize> = vec![0, 1, 1, 2, 2, 4, 5];
    let mut builder = SparseBuilder::multiset(6, source.len());
    assert!(builder.is_multiset(), "Builder created with multiset() is not a multiset");
    builder.extend(source.iter().cloned());
    assert!(builder.is_full(), "Full builder is not full");
    let sv = SparseVector::try_from(builder).unwrap();
    multiset_tests(&sv, 6, &source);
}

//-----------------------------------------------------------------------------
