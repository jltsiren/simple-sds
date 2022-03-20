use simple_sds::bit_vector::BitVector;
use simple_sds::ops::BitVec;
use simple_sds::raw_vector::{RawVector, PushRaw};
use simple_sds::serialize::Serialize;

use std::time::Duration;

use rand::Rng;
use rand::distributions::{Bernoulli, Distribution};

//-----------------------------------------------------------------------------

pub fn random_vector(len: usize, density: f64) -> BitVector {
    let mut data = RawVector::with_capacity(len);
    let mut rng = rand::thread_rng();
    if density == 0.5 {
        while data.len() < len {
            unsafe { data.push_int(rng.gen(), 64); }
        }
    }
    else {
        let dist = Bernoulli::new(density).unwrap();
        let mut iter = dist.sample_iter(&mut rng);
        while data.len() < len {
            data.push_bit(iter.next().unwrap());
        }
    }
    let bv = BitVector::from(data);

    // This test assumes that the number of ones is within 6 stdevs of the expected.
    let ones: f64 = bv.count_ones() as f64;
    let expected: f64 = len as f64 * density;
    let stdev: f64 = (len as f64 * density * (1.0 - density)).sqrt();
    assert!(ones >= expected - 6.0 * stdev && ones <= expected + 6.0 * stdev,
        "random_vector({}, {}): unexpected number of ones: {}", len, density, ones);

    bv
}

pub fn generate_rank_queries(n: usize, len: usize) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(n);

    let mut rng = rand::thread_rng();
    for _ in 0..n {
        let value = rng.gen::<usize>() % len;
        result.push(value);
    }

    result
}

pub fn generate_select_queries(n: usize, ones: usize) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(n);

    let mut rng = rand::thread_rng();
    for _ in 0..n {
        let value = rng.gen::<usize>() % ones;
        result.push(value);
    }

    result
}

pub fn generate_select_zero_queries(n: usize, zeros: usize) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(n);

    let mut rng = rand::thread_rng();
    for _ in 0..n {
        let value = rng.gen::<usize>() % zeros;
        result.push(value);
    }

    result
}

//-----------------------------------------------------------------------------

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

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn peak_memory_usage() -> Result<usize, &'static str> {
    Err("No peak_memory_usage implementation for this OS")
}

//-----------------------------------------------------------------------------

pub fn bitvector_size<'a, T: BitVec<'a> + Serialize>(bv: &'a T) -> String {
    let bytes = bv.size_in_bytes();
    let (size, unit) = readable_size(bytes);
    let bpc = (bytes as f64 * 8.0) / (bv.len() as f64);

    format!("{:.3} {} ({:.3} bpc)", size, unit, bpc)
}

pub fn report_results(queries: usize, total: usize, len: usize, duration: Duration) {
    let average = (total as f64) / (queries as f64);
    let normalized = average / (len as f64);
    let ns = (duration.as_nanos() as f64) / (queries as f64);
    println!("Time:     {:.3} seconds ({:.1} ns/query)", duration.as_secs_f64(), ns);
    println!("Average:  {:.0} absolute, {:.6} normalized", average, normalized);
    println!("");
}

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
