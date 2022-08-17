use simple_sds::ops::{Vector, Access, VectorIndex};
use simple_sds::serialize::Serialize;
use simple_sds::wavelet_matrix::WaveletMatrix;
use simple_sds::{internal, serialize};

use std::time::Instant;
use std::{env, process};

use getopts::Options;

//-----------------------------------------------------------------------------

fn main() {
    let config = Config::new();

    println!("Generating a random vector of {}-bit integers of length {}", config.width, config.len);
    let source = internal::random_vector(config.len, config.width);

    println!("Building a WaveletMatrix");
    let start = Instant::now();
    let wm = WaveletMatrix::from(source);
    internal::report_construction(&wm, wm.len(), start.elapsed());

    // FIXME generate queries
    // FIXME execute queries

    internal::report_memory_usage();
}

//-----------------------------------------------------------------------------

pub struct Config {
    pub len: usize,
    pub width: usize,
    pub queries: usize,
    pub chain_mask: usize,
}

impl Config {
    const BIT_LEN: usize = 28;
    const WIDTH: usize = 16;
    const QUERIES: usize = 1_000_000;
    const CHAIN_MASK: usize = 0xFFFF;

    pub fn new() -> Config {
        let args: Vec<String> = env::args().collect();
        let program = args[0].clone();

        let mut opts = Options::new();
        opts.optopt("l", "bit-len", &format!("use vectors of length 2^INT (default {})", Self::BIT_LEN), "INT");
        opts.optopt("w", "width", &format!("use INT-bit items (default {})", Self::WIDTH), "INT");
        opts.optopt("n", "queries", &format!("number of queries (default {})", Self::QUERIES), "INT");
        opts.optflag("h", "help", "print this help");
        let matches = match opts.parse(&args[1..]) {
            Ok(m) => m,
            Err(f) => {
                eprintln!("{}", f.to_string());
                process::exit(1);
            }
        };

        let mut config = Config {
            len: 1 << Self::BIT_LEN,
            width: Self::WIDTH,
            queries: Self::QUERIES,
            chain_mask: Self::CHAIN_MASK,
        };
        if matches.opt_present("h") {
            let header = format!("Usage: {} [options]", program);
            print!("{}", opts.usage(&header));
            process::exit(0);
        }
        if let Some(s) = matches.opt_str("l") {
            match s.parse::<usize>() {
                Ok(n) => {
                    if n > 63 {
                        eprintln!("Invalid bit length: {}", n);
                        process::exit(1);
                    }
                    config.len = 1 << n;
                },
                Err(f) => {
                    eprintln!("--bit-len: {}", f.to_string());
                    process::exit(1);
                },
            }
        }
        if let Some(s) = matches.opt_str("w") {
            match s.parse::<usize>() {
                Ok(n) => {
                    if n == 0 || n > 64 {
                        eprintln!("Invalid width: {}", n);
                        process::exit(1);
                    }
                    config.width = n;
                },
                Err(f) => {
                    eprintln!("--width: {}", f.to_string());
                    process::exit(1);
                },
            }
        }
        if let Some(s) = matches.opt_str("n") {
            match s.parse::<usize>() {
                Ok(n) => {
                    if n == 0 {
                        eprintln!("Invalid query count: {}", n);
                        process::exit(1);
                    }
                    config.queries = n;
                },
                Err(f) => {
                    eprintln!("--queries: {}", f.to_string());
                    process::exit(1);
                },
            }
        }

        config
    }
}

//-----------------------------------------------------------------------------

pub fn vector_size<T: Vector + Serialize>(v: &T) -> String {
    let bytes = v.size_in_bytes();
    let (size, unit) = internal::readable_size(bytes);
    let bpc = (bytes as f64 * 8.0) / (v.len() as f64);

    format!("{:.3} {} ({:.3} bpc)", size, unit, bpc)
}

//-----------------------------------------------------------------------------
