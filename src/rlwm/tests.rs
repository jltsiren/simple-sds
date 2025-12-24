use super::*;

use crate::{serialize, internal};

//-----------------------------------------------------------------------------

// From + check_vector

#[test]
fn empty_rlwm() {
    let runs = Vec::new();
    let truth = internal::runs_to_values(&runs);
    let rlwm = RLWM::from(runs);
    internal::check_vector(&rlwm, &truth, 1);

}

#[test]
fn rlwm_from_runs() {
    let width = 5;
    let runs = internal::random_integer_runs(51, width, 0.03);
    let truth = internal::runs_to_values(&runs);
    let rlwm = RLWM::from(runs);
    internal::check_vector(&rlwm, &truth, width);
}

#[test]
#[ignore]
fn large_rlwm() {
    let width = 10;
    let runs = internal::random_integer_runs(13842, width, 0.032);
    let truth = internal::runs_to_values(&runs);
    let rlwm = RLWM::from(runs);
    internal::check_vector(&rlwm, &truth, width);
}

//-----------------------------------------------------------------------------

// Serialize

#[test]
fn serialize_empty_rlwm() {
    let runs = Vec::new();
    let rlwm = RLWM::from(runs);
    serialize::test(&rlwm, "rl-wavelet-matrix", None, true);
}

#[test]
fn serialize_rlwm() {
    let runs = internal::random_integer_runs(42, 7, 0.08);
    let rlwm = RLWM::from(runs);
    serialize::test(&rlwm, "rl-wavelet-matrix", None, true);
}

#[test]
#[ignore]
fn serialize_large_rlwm() {
    let runs = internal::random_integer_runs(9836, 11, 0.025);
    let rlwm = RLWM::from(runs);
    serialize::test(&rlwm, "rl-wavelet-matrix", None, true);
}

//-----------------------------------------------------------------------------

// VectorIndex

#[test]
fn empty_rlwm_index() {
    let width = 1;
    let runs = Vec::new();
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

#[test]
fn rlwm_index() {
    let width = 4;
    let runs = internal::random_integer_runs(49, width, 0.04);
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

#[test]
fn rlwm_index_missing_values() {
    let width = 9;
    let runs = internal::random_integer_runs(38, width, 0.07);
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

#[test]
#[ignore]
fn large_rlwm_index() {
    let width = 6;
    let runs = internal::random_integer_runs(1011, width, 0.04);
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

//-----------------------------------------------------------------------------

fn consume_first_last(iter: &AccessIter<'_>, true_index: usize, true_value: u64, true_run_len: usize, len: usize) {
    let mut iter = iter.clone();
    let mut copy = iter.clone();
    let _ = iter.next(); // consume first
    let _ = iter.next_back(); // consume last
    let run = iter.next_run();
    let _ = copy.next_run(); // consume run
    let _ = copy.next_back(); // consume last
    let next = copy.next_run();

    if true_index + true_run_len == len {
        // We are at the last run.
        if true_run_len <= 2 {
            assert!(run.is_none(), "Iterator not exhausted after consuming the ends of a short last run");
        } else {
            let (index, value, run_len) = run.unwrap();
            assert_eq!(index, true_index + 1, "Wrong iterator run index after consuming the ends of a short last run");
            assert_eq!(value, true_value, "Wrong iterator run value after consuming the ends of a short last run");
            assert_eq!(run_len, true_run_len - 2, "Wrong iterator run length after consuming the ends of a short last run");
        }
    } else if true_run_len == 1 {
        // We expect to get the next run.
        if true_index + true_run_len + 1 == len {
            // And it is the final run consisting of a single item.
            assert!(run.is_none(), "Iterator not exhausted after consuming two final single-item runs");
        } else {
            assert_eq!(run, next, "Iterator did not return the next run after consuming the only item at index {}", true_index);
        }
    } else {
        // We have consumed the first item from the current run.
        assert!(run.is_some(), "Iterator exhausted after consuming the first item at index {}", true_index);
        let (index, value, run_len) = run.unwrap();
        assert_eq!(index, true_index + 1, "Wrong iterator run index after consuming the first item at index {}", true_index);
        assert_eq!(value, true_value, "Wrong iterator run value after consuming the first item at index {}", true_index);
        assert_eq!(run_len, true_run_len - 1, "Wrong iterator run length after consuming the first item at index {}", true_index);
    }
}

fn consume_first(iter: &ValueIter<'_>, true_rank: usize, true_index: usize, true_value: usize, true_run_len: usize) {
    let mut iter = iter.clone();
    let mut copy = iter.clone();
    let _ = iter.next(); // consume first
    let run = iter.next_run();
    let _ = copy.next_run(); // consume run
    let next = copy.next_run();

    if true_run_len == 1 {
        assert_eq!(run, next, "Value iterator did not return the next run after consuming the only item of a run for value {}", true_value);
    } else {
        let (rank, index, run_len) = run.unwrap();
        assert_eq!(rank, true_rank + 1, "Wrong value iterator run rank after consuming the first item of a run for value {}", true_value);
        assert_eq!(index, true_index + 1, "Wrong value iterator run index after consuming the first item of a run for value {}", true_value);
        assert_eq!(run_len, true_run_len - 1, "Wrong value iterator run length after consuming the first item of a run for value {}", true_value);
    }
}

// value_iter by consuming first
fn check_runs(rlwm: &RLWM, runs: &[(u64, usize)], width: usize) {
    let mut iter = rlwm.iter();
    let mut value_iters = Vec::with_capacity(1 << width);
    for value in 0..(1 << width) {
        value_iters.push(rlwm.value_iter(value));
    }
    let mut value_ranks = vec![0; 1 << width];

    let mut true_index = 0;
    for &(true_value, true_run_len) in runs.iter() {
        // get_run at each index
        for offset in 0..true_run_len {
            let (value, run_len) = rlwm.get_run(true_index + offset);
            assert_eq!(value, true_value, "Wrong get_run value at index {}", true_index + offset);
            assert_eq!(run_len, true_run_len - offset, "Wrong get_run run_len at index {}", true_index + offset);
        }

        // next_run from AccessIter
        consume_first_last(&iter, true_index, true_value, true_run_len, rlwm.len());
        let run = iter.next_run();
        assert!(run.is_some(), "Iterator exhausted too early at index {}", true_index);
        let (index, value, run_len) = run.unwrap();
        assert_eq!(index, true_index, "Wrong iterator run index at index {}", true_index);
        assert_eq!(value, true_value, "Wrong iterator run value at index {}", true_index);
        assert_eq!(run_len, true_run_len, "Wrong iterator run length at index {}", true_index);

        // next_run from ValueIter
        let val = true_value as usize;
        consume_first(&value_iters[val], value_ranks[val], true_index, val, true_run_len);
        let run = value_iters[val].next_run();
        assert!(run.is_some(), "Value iterator exhausted too early for value {} at index {}", true_value, true_index);
        let (rank, index, run_len) = run.unwrap();
        assert_eq!(rank, value_ranks[val], "Wrong value iterator run rank for value {} at index {}", true_value, true_index);
        assert_eq!(index, true_index, "Wrong value iterator run index for value {} at index {}", true_value, true_index);
        assert_eq!(run_len, true_run_len, "Wrong value iterator run length for value {} at index {}", true_value, true_index);

        // select_run for each rank within the run
        for offset in 0..true_run_len {
            let rank = value_ranks[val] + offset;
            let run = rlwm.select_run(rank, true_value);
            assert!(run.is_some(), "select_run returned None for value {}, rank {}", true_value, rank);
            let (index, run_len) = run.unwrap();
            assert_eq!(index, true_index + offset, "Wrong select_run index for value {}, rank {}", true_value, rank);
            assert_eq!(run_len, true_run_len - offset, "Wrong select_run run length for value {}, rank {}", true_value, rank);
        }

        true_index += true_run_len;
        value_ranks[val] += true_run_len;
    }

    // Check that there are no extra runs or items.
    // get_run will panic if out of bounds, so no need to check that.
    let mut iter_copy = iter.clone();
    assert!(iter.next_run().is_none(), "Iterator has extra runs after end");
    assert!(iter_copy.next().is_none(), "Iterator has extra items after end");
    for value in 0..(1 << width) {
        let val = value as usize;
        let mut iter_copy = value_iters[val].clone();
        assert!(value_iters[val].next_run().is_none(), "Value iterator has extra runs after end for value {}", value);
        assert!(iter_copy.next().is_none(), "Value iterator has extra items after end for value {}", value);
        assert!(rlwm.select_run(value_ranks[val], value).is_none(), "select_run has extra runs after end for value {}", value);
    }
}

#[test]
fn empty_rlwm_runs() {
    let width = 1;
    let runs = Vec::new();
    let rlwm = RLWM::from(runs.clone());
    check_runs(&rlwm, &runs, width);
}

#[test]
fn rlwm_runs() {
    let width = 7;
    let runs = internal::random_integer_runs(128, width, 0.05);
    let rlwm = RLWM::from(runs.clone());
    let runs = internal::maximal_runs(runs);
    check_runs(&rlwm, &runs, width);
}

#[test]
#[ignore]
fn large_rlwm_runs() {
    let width = 10;
    let runs = internal::random_integer_runs(4095, width, 0.03);
    let rlwm = RLWM::from(runs.clone());
    let runs = internal::maximal_runs(runs);
    check_runs(&rlwm, &runs, width);
}

//-----------------------------------------------------------------------------
