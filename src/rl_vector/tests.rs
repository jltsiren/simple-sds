use super::*;

use crate::bit_vector::BitVector;
use crate::{internal, serialize};

use std::cmp;

//-----------------------------------------------------------------------------

fn build_rl_vector(runs: &[(usize, usize)], universe: usize) -> RLVector {
    let mut builder = RLBuilder::new();
    for (start, len) in runs.iter() {
        let _ = builder.try_set(*start, *len).unwrap();
    }
    builder.set_len(universe);
    RLVector::from(builder)
}

fn random_rl_vector(num_runs: usize, p: f64) -> RLVector {
    let (runs, universe) = internal::random_runs(num_runs, p);
    build_rl_vector(&runs, universe)
}

fn zero_vector(len: usize) -> RLVector {
    let mut builder = RLBuilder::new();
    builder.set_len(len);
    RLVector::from(builder)
}

fn one_vector(len: usize) -> RLVector {
    let mut builder = RLBuilder::new();
    builder.try_set(0, len).unwrap();
    RLVector::from(builder)
}

//-----------------------------------------------------------------------------

fn try_iter(rv: &RLVector) {
    internal::try_bitvec_iter(rv);
}

fn try_run_iter(rv: &RLVector, runs: Option<&[(usize, usize)]>) {
    // Compare to the true runs.
    if let Some(runs) = runs {
        let found: Vec<(usize, usize)> = rv.run_iter().collect();
        assert_eq!(found.len(), runs.len(), "Found an invalid number of runs");
        for (index, run) in rv.run_iter().enumerate() {
            assert_eq!(run, runs[index], "Invalid run {}", index);
        }
    }

    // Check that the runs are present in the bitvector.
    let mut visited = 0;
    for (start, len) in rv.run_iter() {
        for pos in start..(start + len) {
            assert!(rv.get(pos), "Bit {} in run {}..{} was not set (forward)", pos, start, start + len);
        }
        visited += len;
    }
    assert_eq!(visited, rv.count_ones(), "Iter did not visit all set bits");
}

#[test]
fn empty_vector() {
    let empty = zero_vector(0);
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.count_ones(), 0, "Empty vector contains ones");
    assert_eq!(empty.count_zeros(), 0, "Empty vector contains zeros");
    assert!(empty.iter().next().is_none(), "Non-empty iterator from an empty vector");
}

#[test]
fn construction_methods() {
    let (runs, universe) = internal::random_runs(52, 0.02);
    let ones = runs.iter().fold(0, |acc, x| acc + x.1);

    let full_runs = build_rl_vector(&runs, universe);
    assert!(!full_runs.is_empty(), "The bitvector is empty");
    assert_eq!(full_runs.len(), universe, "Invalid length");
    assert_eq!(full_runs.count_ones(), ones, "Invalid number of ones");
    assert_eq!(full_runs.count_zeros(), universe - ones, "Invalid number of zeros");

    {
        let mut builder = RLBuilder::new();
        for (start, len) in runs.iter() {
            let mut offset = 0;
            while offset < *len {
                let l = cmp::min(len - offset, 3);
                let _ = builder.try_set(start + offset, l).unwrap();
                offset += l;
            }
        }
        builder.set_len(universe);
        let partial_runs = RLVector::from(builder);
        assert_eq!(partial_runs, full_runs, "Bitvector built from partial runs is incorrect");
    }

    {
        let mut builder = RLBuilder::new();
        for (start, len) in runs.iter() {
            for offset in 0..*len {
                let _ = builder.try_set(start + offset, 1).unwrap();
            }
        }
        builder.set_len(universe);
        let bit_by_bit = RLVector::from(builder);
        assert_eq!(bit_by_bit, full_runs, "Bitvector built bit-by-bit is incorrect");
    }
}

#[test]
fn conversions() {
    let original = random_rl_vector(40, 0.08);
    let bv = BitVector::copy_bit_vec(&original);
    let copy = RLVector::copy_bit_vec(&bv);
    assert_eq!(copy, original, "Conversions changed the contents of the RLVector");
}

#[test]
fn uniform_vector() {
    let zeros = zero_vector(1766);
    assert!(!zeros.is_empty(), "The zero vector is empty");
    assert_eq!(zeros.len(), 1766, "Invalid length for the zero vector");
    assert_eq!(zeros.count_ones(), 0, "Invalid number of ones in the zero vector");
    assert_eq!(zeros.count_zeros(), zeros.len(), "Invalid number of zeros in the zero vector");
    assert_eq!(zeros.iter().len(), zeros.len(), "Invalid size hint from the zero vector");
    assert_eq!(zeros.iter().filter(|b| !*b).count(), zeros.len(), "Some bits were set in the iterator");

    let ones = one_vector(2201);
    assert!(!ones.is_empty(), "The ones vector is empty");
    assert_eq!(ones.len(), 2201, "Invalid length for the ones vector");
    assert_eq!(ones.count_ones(), ones.len(), "Invalid number of ones in the ones vector");
    assert_eq!(ones.count_zeros(), 0, "Invalid number of zeros in the ones vector");
    assert_eq!(ones.iter().len(), ones.len(), "Invalid size hint from the ones vector");
    assert_eq!(ones.iter().filter(|b| *b).count(), ones.len(), "Some bits were unset in the iterator");
}

#[test]
fn access() {
    let (runs, universe) = internal::random_runs(49, 0.025);
    let rv = build_rl_vector(&runs, universe);
    assert_eq!(rv.len(), universe, "Invalid bitvector length");

    let mut iter = runs.iter().copied();
    let mut next = iter.next();
    for i in 0..rv.len() {
        match next {
            Some((start, len)) => {
                if i >= start + len {
                    next = iter.next();
                    assert!(!rv.get(i), "Set bit {} just after run {}..{}", i, start, start + len);
                } else if i >= start {
                    assert!(rv.get(i), "Unset bit {} within run {}..{}", i, start, start + len);
                } else {
                    assert!(!rv.get(i), "Set bit {} before run {}..{}", i, start, start + len);
                }
            },
            None => {
                assert!(!rv.get(i), "Set bit {} after the final run", i);
            },
        }
    }
}

#[test]
fn iter() {
    let rv = random_rl_vector(69, 0.03);
    try_iter(&rv);
}

#[test]
fn run_iter() {
    let (runs, universe) = internal::random_runs(72, 0.025);
    let rv = build_rl_vector(&runs, universe);
    try_run_iter(&rv, Some(&runs));
}

#[test]
fn serialize() {
    let rv = random_rl_vector(49, 0.01);
    let _ = serialize::test(&rv, "rl-vector", None, true);
}

#[test]
#[ignore]
fn large() {
    let rv = random_rl_vector(19379, 0.025);
    try_iter(&rv);
    let _ = serialize::test(&rv, "large-rl-vector", None, true);
}

#[test]
#[ignore]
fn large_runs() {
    let (runs, universe) = internal::random_runs(21132, 0.04);
    let rv = build_rl_vector(&runs, universe);
    try_run_iter(&rv, Some(&runs));
}

//-----------------------------------------------------------------------------

// FIXME try_rank

// FIXME rank: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------

// FIXME try_select, try_one_iter

// FIXME select: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------

// FIXME try_select_zero, try_zero_iter

// FIXME select_zero: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------

// FIXME try_pred_succ

// FIXME pred_succ: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------
