// Utility functions for tests.

use crate::ops::{Vector, Access, VectorIndex, BitVec, Rank, Select, SelectZero, PredSucc};
use crate::serialize::Serialize;
use crate::bits;

use std::time::Duration;

use rand::Rng;
use rand_distr::{Geometric, Distribution};

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

// Returns a vector of positions and universe size.
// The distances between positions are `Geometric(density)`.
pub fn random_positions(n: usize, density: f64) -> (Vec<usize>, usize) {
    let mut positions: Vec<usize> = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let dist = Geometric::new(density).unwrap();
    let mut universe = 0;

    let mut iter = dist.sample_iter(&mut rng);
    while positions.len() < n {
        let pos = universe + (iter.next().unwrap() as usize);
        positions.push(pos);
        universe = pos + 1;
    }
    universe += iter.next().unwrap() as usize;

    (positions, universe)
}

// Returns `n` random (start, length) runs, where gaps and lengths are `Geometric(p)`.
// The second return value is universe size.
// Note that `p` is the flip probability.
pub fn random_runs(n: usize, p: f64) -> (Vec<(usize, usize)>, usize) {
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    let dist = Geometric::new(p).unwrap();

    let start_with_one: bool = rng.gen();
    let end_with_zero: bool = rng.gen();
    let mut universe = 0;
    let mut iter = dist.sample_iter(&mut rng);
    if start_with_one {
        let len = 1 + (iter.next().unwrap() as usize);
        runs.push((0, len));
        universe = len;
    }
    while runs.len() < n {
        let start = universe + 1 + (iter.next().unwrap() as usize);
        let len = 1 + (iter.next().unwrap() as usize);
        runs.push((start, len));
        universe = start + len;
    }
    if end_with_zero {
        universe += 1 + (iter.next().unwrap() as usize);
    }

    (runs, universe)
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

// Tests for `BitVec` and related traits.

// Check that the iterator visits all values correctly.
pub fn try_bitvec_iter<'a, T: BitVec<'a>>(bv: &'a T) {
    assert_eq!(bv.iter().len(), bv.len(), "Invalid Iter length");

    // Forward.
    let mut visited = 0;
    for (index, value) in bv.iter().enumerate() {
        assert_eq!(value, bv.get(index), "Invalid value {} (forward)", index);
        visited += 1;
    }
    assert_eq!(visited, bv.len(), "Iter did not visit all values");
}

// Check that rank queries work correctly at every position.
pub fn try_rank<'a, T: Rank<'a>>(bv: &'a T) {
    assert!(bv.supports_rank(), "Failed to enable rank support");
    assert_eq!(bv.rank(bv.len()), bv.count_ones(), "Invalid rank at vector size");

    let mut rank: usize = 0;
    for i in 0..bv.len() {
        assert_eq!(bv.rank(i), rank, "Invalid rank at {}", i);
        rank += bv.get(i) as usize;
    }
}

// Check that select queries work correctly for every rank.
// Use increment 1 normally and 0 with multisets.
pub fn try_select<'a, T: Select<'a>>(bv: &'a T, increment: usize) {
    assert!(bv.supports_select(), "Failed to enable select support");
    assert!(bv.select(bv.count_ones()).is_none(), "Got a result for select past the end");
    assert!(bv.select_iter(bv.count_ones()).next().is_none(), "Got a result for select_iter past the end");

    let mut next: usize = 0;
    for i in 0..bv.count_ones() {
        let value = bv.select_iter(i).next().unwrap();
        assert_eq!(value.0, i, "Invalid rank for select_iter({})", i);
        assert!(value.1 >= next, "select_iter({}) == {}, expected at least {}", i, value.1, next);
        let index = bv.select(i).unwrap();
        assert_eq!(index, value.1, "Different results for select({}) and select_iter({})", i, i);
        assert!(bv.get(index), "Bit select({}) == {} is not set", i, index);
        next = value.1 + increment;
    }
}

// Check that the one iterator visits all set bits correctly.
// Use increment 1 normally and 0 with multisets.
pub fn try_one_iter<'a, T: Select<'a>>(bv: &'a T, increment: usize) {
    assert_eq!(bv.one_iter().len(), bv.count_ones(), "Invalid OneIter length");

    // Iterate forward.
    let mut next: (usize, usize) = (0, 0);
    for (index, value) in bv.one_iter() {
        assert_eq!(index, next.0, "Invalid rank from OneIter (forward)");
        assert!(value >= next.1, "Too small value from OneIter (forward)");
        assert!(bv.get(value), "OneIter returned an unset bit (forward)");
        next = (next.0 + 1, value + increment);
    }

    assert_eq!(next.0, bv.count_ones(), "OneIter did not visit all unset bits");
}

// Check that select_zero queries work correctly for every rank.
pub fn try_select_zero<'a, T: SelectZero<'a>>(bv: &'a T) {
    assert!(bv.supports_select_zero(), "Failed to enable select_zero support");
    assert!(bv.select_zero(bv.count_zeros()).is_none(), "Got a result for select_zero past the end");
    assert!(bv.select_zero_iter(bv.count_zeros()).next().is_none(), "Got a result for select_zero_iter past the end");

    let mut next: usize = 0;
    for i in 0..bv.count_zeros() {
        let value = bv.select_zero_iter(i).next().unwrap();
        assert_eq!(value.0, i, "Invalid rank for select_zero_iter({})", i);
        assert!(value.1 >= next, "select_zero_iter({}) == {}, expected at least {}", i, value.1, next);
        let index = bv.select_zero(i).unwrap();
        assert_eq!(index, value.1, "Different results for select_zero({}) and select_zero_iter({})", i, i);
        assert!(!bv.get(index), "Bit select_zero({}) == {} is set", i, index);
        next = value.1 + 1;
    }
}

// Check that the zero iterator visits all unset bits correctly.
pub fn try_zero_iter<'a, T: SelectZero<'a>>(bv: &'a T) {
    assert_eq!(bv.zero_iter().len(), bv.count_zeros(), "Invalid ZeroIter length");

    // Iterate forward.
    let mut next: (usize, usize) = (0, 0);
    for (index, value) in bv.zero_iter() {
        assert_eq!(index, next.0, "Invalid rank from ZeroIter (forward)");
        assert!(value >= next.1, "Too small value from ZeroIter (forward)");
        assert!(!bv.get(value), "ZeroIter returned a set bit (forward)");
        next = (next.0 + 1, value + 1);
    }

    assert_eq!(next.0, bv.count_zeros(), "ZeroIter did not visit all unset bits");
}

// Check that predecessor/successor queries work correctly at every position.
pub fn try_pred_succ<'a, T: Rank<'a> + PredSucc<'a>>(bv: &'a T) {
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
                assert!(pred_result.is_none(), "Got a predecessor result before the first set bit");
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
                assert!(succ_result.is_none(), "Got a successor result after the last set bit");
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

    if bv.len() > 0 {
        assert_eq!(bv.predecessor(bv.len()).next(), bv.predecessor(bv.len() - 1).next(), "Invalid predecessor at vector size");
    }
    assert!(bv.successor(bv.len()).next().is_none(), "Invalid successor at vector size");
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
