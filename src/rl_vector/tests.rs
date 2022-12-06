use super::*;

use rand_distr::{Geometric, Distribution};
use rand::Rng;

//-----------------------------------------------------------------------------

// FIXME move to internal
// Returns `n` random (start, length) runs, where gaps and lengths are `Geometric(p)`.
// The second return value is universe size.
// Note that `p` is the flip probability.
fn random_runs(n: usize, p: f64) -> (Vec<(usize, usize)>, usize) {
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

fn random_rl_vector(num_runs: usize, p: f64) -> RLVector {
    let (runs, universe) = random_runs(num_runs, p);

    let mut builder = RLBuilder::new();
    for (start, len) in runs.iter() {
        let _ = builder.try_set(*start, *len).unwrap();
    }
    builder.set_len(universe);

    RLVector::from(builder)
}

// FIXME empty vector, full vector

//-----------------------------------------------------------------------------

// FIXME construction: full runs, partial runs, bit-by-bit

//-----------------------------------------------------------------------------

// FIXME try_iter, try_run_iter

// FIXME basic: empty, non-empty, conversions, uniform, access, iter, serialize, large

//-----------------------------------------------------------------------------

// FIXME try_rank

// FIXME rank: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------

// FIXME try_select, try_one_iter

// FIXME select: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------

// FIXME try_select_zero, try_zero_iter

// FIXME select_zero: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------

// FIXME try_pred_succ

// FIXME pred_succ: empty, non-empty, uniform, large

//-----------------------------------------------------------------------------
