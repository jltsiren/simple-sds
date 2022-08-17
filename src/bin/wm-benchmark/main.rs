use simple_sds::ops::{Vector, Access, VectorIndex};
use simple_sds::serialize::Serialize;
use simple_sds::wavelet_matrix::WaveletMatrix;
use simple_sds::internal;

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

    println!("Generating {} random access queries over the vector", config.queries);
    let access_queries = internal::random_queries(config.queries, wm.len());
    println!("");

    println!("Generating {} random rank queries over the vector", config.queries);
    let rank_queries = query_pairs(config.queries, wm.len(), wm.width());
    println!("");

    println!("Generating {} random inverse select queries over the vector", config.queries);
    let inverse_select_queries = internal::random_queries(config.queries, wm.len());
    println!("");

    println!("Generating {} random select queries over the vector", config.queries);
    let select_queries = query_pairs(config.queries, wm.len() >> wm.width(), wm.width());
    println!("");

    println!("Generating {} random predecessor queries over the vector", config.queries);
    let predecessor_queries = query_pairs(config.queries, wm.len(), wm.width());
    println!("");

    println!("Generating {} random successor queries over the vector", config.queries);
    let successor_queries = query_pairs(config.queries, wm.len(), wm.width());
    println!("");

    independent_access(&wm, &access_queries, "WaveletMatrix");
    independent_rank(&wm, &rank_queries, "WaveletMatrix");
    independent_inverse_select(&wm, &inverse_select_queries, "WaveletMatrix");
    independent_select(&wm, &select_queries, "WaveletMatrix");
    independent_predecessor(&wm, &predecessor_queries, "WaveletMatrix");
    independent_successor(&wm, &successor_queries, "WaveletMatrix");

    internal::report_memory_usage();
}

//-----------------------------------------------------------------------------

pub struct Config {
    pub len: usize,
    pub width: usize,
    pub queries: usize,
}

impl Config {
    const BIT_LEN: usize = 28;
    const WIDTH: usize = 16;
    const QUERIES: usize = 1_000_000;

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

pub fn query_pairs(n: usize, universe: usize, alphabet_width: usize) -> Vec<(usize, u64)> {
    let positions = internal::random_queries(n, universe);
    let symbols = internal::random_vector(n, alphabet_width);
    positions.iter().cloned().zip(symbols.iter().cloned()).collect()
}

//-----------------------------------------------------------------------------

fn independent_access<'a, T: Vector<Item = u64> + Access<'a>>(v: &'a T, queries: &[usize], vector_type: &str) {
    println!("{} with {} independent access queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for index in queries {
        let result = v.get(*index);
        total += result;
    }
    let elapsed = now.elapsed();
    internal::report_results(queries.len(), total as usize, 1 << v.width(), elapsed);
}

fn independent_rank<'a, T: Vector<Item = u64> + VectorIndex<'a>>(v: &'a T, queries: &[(usize, u64)], vector_type: &str) {
    println!("{} with {} independent rank queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for (index, value) in queries {
        let result = v.rank(*index, *value);
        total += result;
    }
    let elapsed = now.elapsed();
    internal::report_results(queries.len(), total, v.len() >> v.width(), elapsed);
}

fn independent_inverse_select<'a, T: Vector<Item = u64> + VectorIndex<'a>>(v: &'a T, queries: &[usize], vector_type: &str) {
    println!("{} with {} independent inverse select queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for index in queries {
        let result = v.inverse_select(*index).unwrap_or((v.len() >> v.width(), 0));
        total += result.0;
    }
    let elapsed = now.elapsed();
    internal::report_results(queries.len(), total, v.len() >> v.width(), elapsed);
}

fn independent_select<'a, T: Vector<Item = u64> + VectorIndex<'a>>(v: &'a T, queries: &[(usize, u64)], vector_type: &str) {
    println!("{} with {} independent select queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for (index, value) in queries {
        let result = v.select(*index, *value).unwrap_or(v.len());
        total += result;
    }
    let elapsed = now.elapsed();
    internal::report_results(queries.len(), total, v.len(), elapsed);
}

fn independent_predecessor<'a, T: Vector<Item = u64> + VectorIndex<'a>>(v: &'a T, queries: &[(usize, u64)], vector_type: &str) {
    println!("{} with {} independent predecessor queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for (index, value) in queries {
        let result = v.predecessor(*index, *value).next().unwrap_or((0, 0));
        total += result.1;
    }
    let elapsed = now.elapsed();
    internal::report_results(queries.len(), total, v.len(), elapsed);
}

fn independent_successor<'a, T: Vector<Item = u64> + VectorIndex<'a>>(v: &'a T, queries: &[(usize, u64)], vector_type: &str) {
    println!("{} with {} independent successor queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for (index, value) in queries {
        let result = v.successor(*index, *value).next().unwrap_or((v.len() >> v.width(), v.len()));
        total += result.1;
    }
    let elapsed = now.elapsed();
    internal::report_results(queries.len(), total, v.len(), elapsed);
}

//-----------------------------------------------------------------------------
