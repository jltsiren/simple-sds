use simple_sds::ops::{BitVec, Rank, Select, SelectZero};
use simple_sds::serialize::Serialize;
use simple_sds::bit_vector::BitVector;
use simple_sds::rl_vector::RLVector;
use simple_sds::sparse_vector::SparseVector;
use simple_sds::{internal, serialize};

use std::time::Instant;
use std::{env, process};

use getopts::Options;

mod utils;

//-----------------------------------------------------------------------------

// TODO: More operations: get, one_iter (forward/backward)

fn main() {
    let config = Config::new();

    let mut bv: BitVector = if let Some(basename) = config.infile.as_ref() {
        let bv_file: String = format!("{}.bv", basename);
        println!("Loading BitVector from {}", bv_file);
        serialize::load_from(bv_file).unwrap()
    } else if config.runs {
        println!("Generating a random 2^{}-bit BitVector with flip probability {}", config.bit_len, config.density);
        utils::random_vector_runs(1usize << config.bit_len, config.density)
    } else {
        println!("Generating a random 2^{}-bit BitVector with density {}", config.bit_len, config.density);
        utils::random_vector(1usize << config.bit_len, config.density)
    };
    println!("Ones:     {} (density {:.6})", bv.count_ones(), (bv.count_ones() as f64) / (bv.len() as f64));
    bv.enable_rank();
    bv.enable_select();
    bv.enable_select_zero();

    println!("Generating {} random rank queries over the bitvector", config.queries);
    let rank_queries = internal::random_queries(config.queries, bv.len());
    println!("");

    println!("Generating {} random select queries over the bitvector", config.queries);
    let select_queries = internal::random_queries(config.queries, bv.count_ones());
    println!("");

    println!("Generating {} random select_zero queries over the bitvector", config.queries);
    let select_zero_queries = internal::random_queries(config.queries, bv.count_zeros());
    println!("");

    run_tests(&bv, &rank_queries, &select_queries, &select_zero_queries, "BitVector", "bv", &config);

    let sv: SparseVector = if let Some(basename) = config.infile.as_ref() {
        let sv_file: String = format!("{}.sv", basename);
        println!("Loading SparseVector from {}", sv_file);
        serialize::load_from(sv_file).unwrap()
    } else {
        println!("Creating a SparseVector");
        SparseVector::copy_bit_vec(&bv)
    };
    run_tests(&sv, &rank_queries, &select_queries, &select_zero_queries, "SparseVector", "sv", &config);

    let rv: RLVector = if let Some(basename) = config.infile.as_ref() {
        let rv_file: String = format!("{}.rv", basename);
        println!("Loading RLVector from {}", rv_file);
        serialize::load_from(rv_file).unwrap()
    } else {
        println!("Creating an RLVector");
        RLVector::copy_bit_vec(&bv)
    };
    run_tests(&rv, &rank_queries, &select_queries, &select_zero_queries, "RLVector", "rv", &config);

    internal::report_memory_usage();
}

//-----------------------------------------------------------------------------

pub struct Config {
    pub bit_len: usize,
    pub density: f64,
    pub runs: bool,
    pub queries: usize,
    pub chain_mask: usize,
    pub infile: Option<String>,
    pub outfile: Option<String>,
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
        opts.optflag("r", "runs", "generate runs with the density as flip probability");
        opts.optopt("n", "queries", "number of queries (default 10000000)", "INT");
        opts.optopt("L", "load", "load the vectors from NAME.bv and NAME.sv", "NAME");
        opts.optopt("S", "save", "save the vectors to NAME.bv and NAME.sv", "NAME");
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
            runs: false,
            queries: Self::QUERIES,
            chain_mask: Self::CHAIN_MASK,
            infile: None,
            outfile: None,
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
        config.runs = matches.opt_present("r");
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
        config.infile = matches.opt_str("L");
        config.outfile = matches.opt_str("S");

        config
    }
}

//-----------------------------------------------------------------------------

fn run_tests<'a, T>(bv: &'a T, rank_queries: &[usize], select_queries: &[usize], select_zero_queries: &[usize], name: &str, extension: &str, config: &Config)
where T: BitVec<'a> + Rank<'a> + Select<'a> + SelectZero<'a> + Serialize {
    println!("Size:     {}", utils::bitvector_size(bv));
    println!("");

    independent_rank(bv, rank_queries, name);
    chained_rank(bv, rank_queries, config.chain_mask, name);
    independent_select(bv, select_queries, name);
    chained_select(bv, select_queries, config.chain_mask, name);
    independent_select_zero(bv, select_zero_queries, name);
    chained_select_zero(bv, select_zero_queries, config.chain_mask, name);

    if let Some(basename) = config.outfile.as_ref() {
        let filename: String = format!("{}.{}", basename, extension);
        println!("Saving {} to {}", name, filename);
        serialize::serialize_to(bv, &filename).unwrap();
        println!("");
    }
}

fn independent_rank<'a, T: BitVec<'a> + Rank<'a>>(bv: &'a T, queries: &[usize], vector_type: &str) {
    println!("{} with {} independent rank queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.rank(queries[i]);
        total += result;
    }
    internal::report_results(queries.len(), total, bv.count_ones(), now.elapsed());
}

fn chained_rank<'a, T: BitVec<'a> + Rank<'a>>(bv: &'a T, queries: &[usize], chained_query_mask: usize, vector_type: &str) {
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
    internal::report_results(queries.len(), total, bv.count_ones(), now.elapsed());
}

fn independent_select<'a, T: BitVec<'a> + Select<'a>>(bv: &'a T, queries: &[usize], vector_type: &str) {
    println!("{} with {} independent select queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.select(queries[i]).unwrap();
        total += result;
    }
    internal::report_results(queries.len(), total, bv.len(), now.elapsed());
}

fn chained_select<'a, T: BitVec<'a> + Select<'a>>(bv: &'a T, queries: &[usize], chained_query_mask: usize, vector_type: &str) {
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
    internal::report_results(queries.len(), total, bv.len(), now.elapsed());
}

fn independent_select_zero<'a, T: BitVec<'a> + SelectZero<'a>>(bv: &'a T, queries: &[usize], vector_type: &str) {
    println!("{} with {} independent select_zero queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    for i in 0..queries.len() {
        let result = bv.select_zero(queries[i]).unwrap();
        total += result;
    }
    internal::report_results(queries.len(), total, bv.len(), now.elapsed());
}

fn chained_select_zero<'a, T: BitVec<'a> + SelectZero<'a>>(bv: &'a T, queries: &[usize], chained_query_mask: usize, vector_type: &str) {
    println!("{} with {} chained select_zero queries", vector_type, queries.len());
    let now = Instant::now();
    let mut total = 0;
    let mut prev: usize = 0;
    for i in 0..queries.len() {
        let query = (queries[i] ^ prev) % bv.count_zeros();
        let result = bv.select_zero(query).unwrap();
        total += result;
        prev = result & chained_query_mask;
    }
    internal::report_results(queries.len(), total, bv.len(), now.elapsed());
}

//-----------------------------------------------------------------------------
