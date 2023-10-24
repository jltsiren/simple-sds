use std::cmp;

use simple_sds::bit_vector::BitVector;
use simple_sds::ops::BitVec;
use simple_sds::raw_vector::{RawVector, PushRaw};
use simple_sds::serialize::Serialize;
use simple_sds::internal;

use rand::Rng;
use rand::distributions::Bernoulli;
use rand_distr::{Geometric, Distribution};

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

pub fn random_vector_runs(len: usize, flip: f64) -> BitVector {
    let mut data = RawVector::with_capacity(len);
    let mut rng = rand::thread_rng();
    let dist = Geometric::new(flip).unwrap();

    let mut value: bool = rng.gen();
    let mut iter = dist.sample_iter(&mut rng);
    while data.len() < len {
        let run = cmp::min(1 + (iter.next().unwrap() as usize), len - data.len());
        for _ in 0..run {
            data.push_bit(value);
        }
        value = !value;
    }

    BitVector::from(data)
}

//-----------------------------------------------------------------------------

pub fn bitvector_size<'a, T: BitVec<'a> + Serialize>(bv: &'a T) -> String {
    let bytes = bv.size_in_bytes();
    let (size, unit) = internal::readable_size(bytes);
    let bpc = (bytes as f64 * 8.0) / (bv.len() as f64);

    format!("{:.3} {} ({:.3} bpc)", size, unit, bpc)
}

//-----------------------------------------------------------------------------
