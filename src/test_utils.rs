// Utility functions for tests.

use crate::ops::{Vector, Access, VectorIndex};
use crate::bits;

use rand::Rng;

//-----------------------------------------------------------------------------

// Returns a vector of `len` random `width`-bit integers.
pub fn random_vector(len: usize, width: usize) -> Vec<u64> {
    let mut result: Vec<u64> = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..len {
        let value: u64 = rng.gen();
        result.push(value & bits::low_set(width));
    }
    result
}

//-----------------------------------------------------------------------------

// Check that the vector is equal to the truth vector.
pub fn check_vector<'a, T>(v: &'a T, truth: &[u64], width: usize)
where
    T: Vector<Item = u64> + Access<'a> + Clone + IntoIterator<Item = u64>,
    <T as Access<'a>>::Iter: DoubleEndedIterator,

{
    assert_eq!(v.len(), truth.len(), "Invalid vector length");
    assert_eq!(v.is_empty(), truth.is_empty(), "Invalid vector emptiness");
    assert_eq!(v.width(), width, "Invalid vector width");

    for i in 0..v.len() {
        assert_eq!(v.get(i), truth[i], "Invalid value {}", i);
    }
    assert!(v.iter().eq(truth.iter().cloned()), "Invalid iterator (forward)");

    let mut index = v.len();
    let mut iter = v.iter();
    while let Some(value) = iter.next_back() {
        index -= 1;
        assert_eq!(value, truth[index], "Invalid value {} (backward)", index);
    }

    // Meet in the middle.
    let mut next = 0;
    let mut limit = v.len();
    let mut iter = v.iter();
    while next < limit {
        assert_eq!(iter.next(), Some(truth[next]), "Invalid value {} (forward, meet in middle)", next);
        next += 1;
        if next >= limit {
            break;
        }
        limit -= 1;
        assert_eq!(iter.next_back(), Some(truth[limit]), "Invalid value {} (backward, meet in middle", limit);
    }
    assert!(iter.next().is_none(), "Got a value from iterator after meeting in the middle");
    assert!(iter.next_back().is_none(), "Got a value (backward) from iterator after meeting in the middle");

    let copy: Vec<u64> = v.clone().into_iter().collect();
    assert_eq!(copy, *truth, "Invalid vector from into_iter()");
}

//-----------------------------------------------------------------------------

// Tests for `VectorIndex`.

// Test `contains`.
pub fn check_contains<'a, T>(v: &'a T, width: usize)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    for value in 0..(1 << width) {
        let should_have = v.iter().any(|x| x == value);
        assert_eq!(v.contains(value), should_have, "Invalid contains({})", value);
    }
}

// Test `rank`.
pub fn check_rank<'a, T>(v: &'a T, width: usize)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    for value in 0..(1 << width) {
        let mut count = 0;
        for index in 0..=v.len() {
            assert_eq!(v.rank(index, value), count, "Invalid rank({}, {})", index, value);
            if index < v.len() && v.get(index) == value {
                count += 1;
            }
        }
    }
}

// Test `inverse_select`.
pub fn check_inverse_select<'a, T>(v: &'a T)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    for i in 0..v.len() {
        let result = v.inverse_select(i);
        assert!(result.is_some(), "No result for inverse_select({})", i);
        let result = result.unwrap();
        assert_eq!(v.select(result.0, result.1), Some(i), "Invalid inverse_select({})", i);
    }
    assert!(v.inverse_select(v.len()).is_none(), "Got an inverse_select() result past the end");
}

// Test `value_iter`.
pub fn check_value_iter<'a, T>(v: &'a T, width: usize)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    for value in 0..(1 << width) {
        let mut iter = v.value_iter(value);
        assert_eq!(T::value_of(&iter), value, "Invalid value for value_iter({})", value);
        let mut rank = 0;
        let mut index = 0;
        while index < v.len() {
            if v.get(index) == value {
                assert_eq!(iter.next(), Some((rank, index)), "Invalid result of rank {} from value_iter({})", rank, value);
                rank += 1;
            }
            index += 1;
        }
        assert!(iter.next().is_none(), "Got a past-the-end result from value_iter({})", value);
    }
}

// Test `select` and `select_iter`.
pub fn check_select<'a, T>(v: &'a T, width: usize)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    for value in 0..(1 << width) {
        let mut rank = 0;
        let mut index = 0;
        while index < v.len() {
            if v.get(index) == value {
                assert_eq!(v.select(rank, value), Some(index), "Invalid select({}, {})", rank, value);
                assert_eq!(v.select_iter(rank, value).next(), Some((rank, index)), "Invalid select_iter({}, {})", rank, value);
                rank += 1;
            }
            index += 1;
        }
        assert!(v.select(rank, value).is_none(), "Got a past-the-end result from select({}, {})", rank, value);
        assert!(v.select_iter(rank, value).next().is_none(), "Got a past-the-end result from select_iter({}, {})", rank, value);
    }
}

// Test `predecessor` and `successor`.
pub fn check_pred_succ<'a, T>(v: &'a T, width: usize)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    for value in 0..(1 << width) {
        let mut iter = v.value_iter(value);
        let mut prev: Option<(usize, usize)> = None;
        let mut next: Option<(usize, usize)> = iter.next();

        // Try also querying at past-the-end position.
        for index in 0..=v.len() {
            if next.is_some() && index == next.unwrap().1 {
                assert_eq!(v.predecessor(index, value).next(), next, "Invalid predecessor({}, {}) at occurrence", index, value);
                assert_eq!(v.successor(index, value).next(), next, "Invalid successor({}, {}) at occurrence", index, value);
                prev = next;
                next = iter.next();
            } else {
                assert_eq!(v.predecessor(index, value).next(), prev, "Invalid predecessor({}, {})", index, value);
                assert_eq!(v.successor(index, value).next(), next, "Invalid successor({}, {})", index, value);
            }
        }
    }
}

pub fn check_vector_index<'a, T>(v: &'a T, width: usize)
where
    T: Vector<Item = u64> + Access<'a> + VectorIndex<'a>,
{
    check_contains(v, width);
    check_rank(v, width);
    check_inverse_select(v);
    check_value_iter(v, width);
    check_select(v, width);
    check_pred_succ(v, width);
}

//-----------------------------------------------------------------------------
