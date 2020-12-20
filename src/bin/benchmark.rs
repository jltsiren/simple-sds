use simple_sds::bit_vector::BitVector;
use simple_sds::ops::{BitVec, Rank, Select};
use simple_sds::raw_vector::{RawVector, PushRaw};
use simple_sds::serialize::Serialize;
use simple_sds::bits;

use std::time::{Duration, Instant};

use rand::Rng;
use rand::distributions::{Bernoulli, Distribution};

//-----------------------------------------------------------------------------

// TODO: More general benchmarks
// Parameters: length density, queries
// More operations: get, select_zero, one_iter (forward/backward)
// Consistent output format
// Memory usage

fn main() {
    let bit_len: usize = 32;
    let density: f64 = 0.5;
    let num_queries: usize = 10_000_000;
    let chained_query_mask: usize = 0xFFFF;

    println!("Generating a random 2^{}-bit vector with density {}", bit_len, density);
    let mut bv = random_vector(1usize << bit_len, density);
    println!("Ones: {}", bv.count_ones());
    println!("Size: {}", bitvector_size(&bv));
    println!("");

    println!("Enabling rank support");
    bv.enable_rank();
    println!("Size: {}", bitvector_size(&bv));
    println!("");

    println!("Enabling select support");
    bv.enable_select();
    println!("Size: {}", bitvector_size(&bv));
    println!("");

    println!("Generating {} random rank queries over the bitvector", num_queries);
    let queries = generate_rank_queries(bit_len, num_queries);
    println!("");

    println!("Running {} independent rank queries", queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.rank(queries[i]);
        total += result;
    }
    let duration = now.elapsed();
    let average = (total as f64) / (queries.len() as f64);
    println!("Time: {}", query_time(num_queries, &duration));
    println!("Average rank: {}", average);
    println!("");

    println!("Running {} chained rank queries", queries.len());
    let now = Instant::now();
    let mut total = 0;
    let mut prev: usize = 0;
    for i in 0..queries.len() {
        let query = queries[i] ^ prev;
        let result = bv.rank(query);
        total += result;
        prev = result & chained_query_mask;
    }
    let duration = now.elapsed();
    let average = (total as f64) / (queries.len() as f64);
    println!("Time: {}", query_time(num_queries, &duration));
    println!("Average rank: {}", average);
    println!("");

    println!("Generating {} random select queries over the bitvector", num_queries);
    let queries = generate_select_queries(bv.count_ones(), num_queries);
    println!("");

    println!("Running {} independent select queries", queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.select(queries[i]).next().unwrap().1;
        total += result;
    }
    let duration = now.elapsed();
    let average = (total as f64) / (queries.len() as f64);
    println!("Time: {}", query_time(num_queries, &duration));
    println!("Average value: {}", average);
    println!("");

    println!("Running {} chained select queries", queries.len());
    let now = Instant::now();
    let mut total = 0;
    let mut prev: usize = 0;
    for i in 0..queries.len() {
        let query = (queries[i] ^ prev) % bv.count_ones();
        let result = bv.select(query).next().unwrap().1;
        total += result;
        prev = result & chained_query_mask;
    }
    let duration = now.elapsed();
    let average = (total as f64) / (queries.len() as f64);
    println!("Time: {}", query_time(num_queries, &duration));
    println!("Average value: {}", average);
    println!("");
}

//-----------------------------------------------------------------------------

fn random_vector(len: usize, density: f64) -> BitVector {
    let mut data = RawVector::with_capacity(len);
    let mut rng = rand::thread_rng();
    if density == 0.5 {
        while data.len() < len {
            data.push_int(rng.gen(), 64);
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

fn generate_rank_queries(bit_len: usize, n: usize) -> Vec<usize> {
    let len = 1usize << bit_len;
    let mut result: Vec<usize> = Vec::with_capacity(len);

    let mut rng = rand::thread_rng();
    for _ in 0..n {
        let value = rng.gen::<u64>() & bits::low_set(bit_len);
        result.push(value as usize);
    }

    result
}

fn generate_select_queries(ones: usize, n: usize) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(ones);

    let mut rng = rand::thread_rng();
    for _ in 0..n {
        let value = rng.gen::<usize>() % ones;
        result.push(value);
    }

    result
}

fn bitvector_size(bv: &BitVector) -> String {
    format!("{} bytes ({} bpc)", bv.size_in_bytes(), (bv.size_in_bytes() as f64 * 8.0) / (bv.len() as f64))
}

fn query_time(n: usize, duration: &Duration) -> String {
    let ns = (duration.as_nanos() as f64) / (n as f64);
    format!("{} seconds ({} ns / query)", duration.as_secs_f64(), ns)
}

//-----------------------------------------------------------------------------
