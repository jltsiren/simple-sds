// Utility functions for tests.

use crate::ops::{Vector, Access, VectorIndex};
use crate::serialize::Serialize;
use crate::bits;

use std::time::Duration;

use rand::Rng;

//-----------------------------------------------------------------------------

// Returns a vector of `len` random `width`-bit integers.
pub fn random_vector(len: usize, width: usize) -> Vec<u64> {
    let mut result: Vec<u64> = Vec::with_capacity(len);
    let mut rng = rand::thread_rng();
    for _ in 0..len {
        let value: u64 = rng.gen();
        result.push(value & bits::low_set(width));
    }
    result
}

// Returns `n` random values in `0..universe`.
pub fn random_queries(n: usize, universe: usize) -> Vec<usize>{
    let mut result: Vec<usize> = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    for _ in 0..n {
        let value: usize = rng.gen();
        result.push(value % universe);
    }
    result
} 

//-----------------------------------------------------------------------------

// Returns a human-readable representation of a size in bytes.
pub fn readable_size(bytes: usize) -> (f64, &'static str) {
    let units: Vec<(f64, &'static str)> = vec![
        (1.0, "B"),
        (1024.0, "KiB"),
        (1024.0 * 1024.0, "MiB"),
        (1024.0 * 1024.0 * 1024.0, "GiB"),
        (1024.0 * 1024.0 * 1024.0 * 1024.0, "TiB"),
    ];

    let value = bytes as f64;
    let mut unit = 0;
    for i in 1..units.len() {
        if value >= units[i].0 {
            unit = i;
        } else {
            break;
        }
    }

    (value / units[unit].0, units[unit].1)
}

// Prints a summary report for construction.
//
// * `object`: The structure that was built.
// * `len`: Length of the object.
// * `duration`: Time used for construction.
pub fn report_construction<T: Serialize>(object: &T, len: usize, duration: Duration) {
    let ns = (duration.as_nanos() as f64) / (len as f64);
    let (size, unit) = readable_size(object.size_in_bytes());
    println!("Time:     {:.3} seconds ({:.1} ns/symbol)", duration.as_secs_f64(), ns);
    println!("Size:     {:.3} {}", size, unit);
    println!("");
}

// Prints a summary report for query results.
//
// * `queries`: Number of queries.
// * `total`: Sum of query results.
// * `len`: Codomain size (size of the range of query results).
// * `duration`: Time used for executing the queries.
pub fn report_results(queries: usize, total: usize, len: usize, duration: Duration) {
    let average = (total as f64) / (queries as f64);
    let normalized = average / (len as f64);
    let ns = (duration.as_nanos() as f64) / (queries as f64);
    println!("Time:     {:.3} seconds ({:.1} ns/query)", duration.as_secs_f64(), ns);
    println!("Average:  {:.0} absolute, {:.6} normalized", average, normalized);
    println!("");
}

//-----------------------------------------------------------------------------

// Returns peak RSS size so far; Linux version.
#[cfg(target_os = "linux")]
pub fn peak_memory_usage() -> Result<usize, &'static str> {
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        let retval = libc::getrusage(libc::RUSAGE_SELF, &mut rusage as *mut _);
        match retval {
            0 => Ok(rusage.ru_maxrss as usize * 1024),
            _ => Err("libc::getrusage call failed"),
        }
    }
}

// Returns peak RSS size so far; macOS version.
#[cfg(target_os = "macos")]
pub fn peak_memory_usage() -> Result<usize, &'static str> {
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        let retval = libc::getrusage(libc::RUSAGE_SELF, &mut rusage as *mut _);
        match retval {
            0 => Ok(rusage.ru_maxrss as usize),
            _ => Err("libc::getrusage call failed"),
        }
    }
}

// Returns peak RSS size so far; generic version.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn peak_memory_usage() -> Result<usize, &'static str> {
    Err("No peak_memory_usage implementation for this OS")
}

// Prints a memory usage report.
pub fn report_memory_usage() {
    match peak_memory_usage() {
        Ok(bytes) => {
            let (size, unit) = readable_size(bytes);
            println!("Peak memory usage: {:.3} {}", size, unit);
        },
        Err(f) => {
            println!("{}", f);
        },
    }
    println!("");
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
