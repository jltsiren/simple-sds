use super::*;

use crate::{serialize, internal};

//-----------------------------------------------------------------------------

#[test]
fn empty_wm() {
    let truth: Vec<u64> = Vec::new();
    let wm = WaveletMatrix::from(truth.clone());
    internal::check_vector(&wm, &truth, 1);

}

macro_rules! test_wm_from {
    ($name:ident, $t:ident) => {
        #[test]
        fn $name() {
            let width = 6;
            let truth = internal::random_vector(289, width);
        
            let source: Vec<$t> = truth.iter().map(|x| *x as $t).collect();
            let wm = WaveletMatrix::from(source);
            internal::check_vector(&wm, &truth, width);
        }
    }
}

test_wm_from!(wm_from_u8, u8);
test_wm_from!(wm_from_u16, u16);
test_wm_from!(wm_from_u32, u32);
test_wm_from!(wm_from_u64, u64);
test_wm_from!(wm_from_usize, usize);

#[test]
#[ignore]
fn large_wm() {
    let width = 11;
    let truth = internal::random_vector(213951, width);
    let wm = WaveletMatrix::from(truth.clone());
    internal::check_vector(&wm, &truth, width);
}

//-----------------------------------------------------------------------------

#[test]
fn serialize_empty_wm() {
    let source: Vec<u64> = Vec::new();
    let wm = WaveletMatrix::from(source);
    serialize::test(&wm, "wavelet-matrix", None, true);
}

#[test]
fn serialize_wm() {
    let source = internal::random_vector(313, 7);
    let wm = WaveletMatrix::from(source);
    serialize::test(&wm, "wavelet-matrix", None, true);
}

#[test]
#[ignore]
fn serialize_large_wm() {
    let width = 12;
    let source = internal::random_vector(197466, width);
    let wm = WaveletMatrix::from(source);
    serialize::test(&wm, "wavelet-matrix", None, true);
}

//-----------------------------------------------------------------------------

#[test]
fn empty_wm_index() {
    let width = 1;
    let source: Vec<u64> = Vec::new();
    let wm = WaveletMatrix::from(source);
    internal::check_vector_index(&wm, width);
}

#[test]
fn wm_index() {
    let width = 6;
    let source = internal::random_vector(288, width);
    let wm = WaveletMatrix::from(source);
    internal::check_vector_index(&wm, width);
}

#[test]
fn wm_index_missing_values() {
    let width = 9;
    let source = internal::random_vector(244, width);
    let wm = WaveletMatrix::from(source);
    internal::check_vector_index(&wm, width);
}

#[test]
#[ignore]
fn large_wm_index() {
    let width = 7;
    let source = internal::random_vector(26451, width);
    let wm = WaveletMatrix::from(source);
    internal::check_vector_index(&wm, width);
}

//-----------------------------------------------------------------------------
