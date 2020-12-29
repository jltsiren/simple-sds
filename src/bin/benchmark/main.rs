use simple_sds::ops::{BitVec, Rank, Select};
use simple_sds::sparse_vector::SparseVector;

use std::time::Instant;
use std::{env, process};

use getopts::Options;

mod utils;

//-----------------------------------------------------------------------------

// TODO: More operations: get, select_zero, one_iter (forward/backward)

fn main() {
    let config = Config::new();

    println!("Generating a random 2^{}-bit BitVector with density {}", config.bit_len, config.density);
    let mut bv = utils::random_vector(1usize << config.bit_len, config.density);
    println!("Ones:     {} (density {:.6})", bv.count_ones(), (bv.count_ones() as f64) / (bv.len() as f64));
    bv.enable_rank();
    bv.enable_select();
    println!("Size:     {}", utils::bitvector_size(&bv));
    println!("");

    println!("Generating {} random rank queries over the bitvector", config.queries);
    let rank_queries = utils::generate_rank_queries(config.queries, config.bit_len);
    println!("");

    println!("Generating {} random select queries over the bitvector", config.queries);
    let select_queries = utils::generate_select_queries(config.queries, bv.count_ones());
    println!("");

    independent_rank(&bv, &rank_queries, "BitVector");
    chained_rank(&bv, &rank_queries, config.chain_mask, "BitVector");
    independent_select(&bv, &select_queries, "BitVector");
    chained_select(&bv, &select_queries, config.chain_mask, "BitVector");

    println!("Creating a SparseVector");
    let sv = SparseVector::copy_bit_vec(&bv);
    println!("Size:     {}", utils::bitvector_size(&sv));
    println!("");

    independent_rank(&sv, &rank_queries, "SparseVector");
    chained_rank(&sv, &rank_queries, config.chain_mask, "SparseVector");
    independent_select(&sv, &select_queries, "SparseVector");
    chained_select(&sv, &select_queries, config.chain_mask, "SparseVector");

    utils::report_memory_usage();
}

//-----------------------------------------------------------------------------

pub struct Config {
    pub bit_len: usize,
    pub density: f64,
    pub queries: usize,
    pub chain_mask: usize,
}

impl Config {
    const BIT_LEN: usize = 32;
    const DENSITY: f64 = 0.5;
    const QUERIES: usize = 10_000_000;
    const CHAIN_MASK: usize = 0xFFFF;

    pub fn new() -> Config {
        let args: Vec<String> = env::args().collect();
        let program = args[0].clone();

        let mut opts = Options::new();
        opts.optopt("l", "bit-len", "use bitvectors of length 2^INT (default 32)", "INT");
        opts.optopt("d", "density", "density of set bits (default 0.5)", "FLOAT");
        opts.optopt("n", "queries", "number of queries (default 10000000)", "INT");
        opts.optflag("h", "help", "print this help");
        let matches = match opts.parse(&args[1..]) {
            Ok(m) => m,
            Err(f) => {
                eprintln!("{}", f.to_string());
                process::exit(1);
            }
        };

        let mut config = Config {
            bit_len: Self::BIT_LEN,
            density: Self::DENSITY,
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
                    config.bit_len = n;
                },
                Err(f) => {
                    eprintln!("--bit-len: {}", f.to_string());
                    process::exit(1);
                },
            }
        }
        if let Some(s) = matches.opt_str("d") {
            match s.parse::<f64>() {
                Ok(n) => {
                    if n < 0.0 || n > 1.0 {
                        eprintln!("Invalid set bit density: {}", n);
                        process::exit(1);
                    }
                    config.density = n;
                }
                Err(f) => {
                    eprintln!("--density: {}", f.to_string());
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

fn independent_rank<'a, T: BitVec<'a> + Rank<'a>>(bv: &'a T, queries: &Vec<usize>, vector_type: &str) {
    println!("{} with {} independent rank queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.rank(queries[i]);
        total += result;
    }
    utils::report_results(queries.len(), total, bv.count_ones(), now.elapsed());
}

fn chained_rank<'a, T: BitVec<'a> + Rank<'a>>(bv: &'a T, queries: &Vec<usize>, chained_query_mask: usize, vector_type: &str) {
    println!("{} with {} chained rank queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    let mut prev: usize = 0;
    for i in 0..queries.len() {
        let query = queries[i] ^ prev;
        let result = bv.rank(query);
        total += result;
        prev = result & chained_query_mask;
    }
    utils::report_results(queries.len(), total, bv.count_ones(), now.elapsed());
}

fn independent_select<'a, T: BitVec<'a> + Select<'a>>(bv: &'a T, queries: &Vec<usize>, vector_type: &str) {
    println!("{} with {} independent select queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.select(queries[i]).unwrap();
        total += result;
    }
    utils::report_results(queries.len(), total, bv.len(), now.elapsed());
}

fn chained_select<'a, T: BitVec<'a> + Select<'a>>(bv: &'a T, queries: &Vec<usize>, chained_query_mask: usize, vector_type: &str) {
    println!("{} with {} chained select queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    let mut prev: usize = 0;
    for i in 0..queries.len() {
        let query = (queries[i] ^ prev) % bv.count_ones();
        let result = bv.select(query).unwrap();
        total += result;
        prev = result & chained_query_mask;
    }
    utils::report_results(queries.len(), total, bv.len(), now.elapsed());
}

//-----------------------------------------------------------------------------
