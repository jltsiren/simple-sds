use super::*;

use crate::{serialize, internal};

//-----------------------------------------------------------------------------

// From + check_vector

#[test]
fn empty_rlwm() {
    let runs = Vec::new();
    let truth = internal::runs_to_values(&runs);
    let rlwm = RLWM::from(runs);
    internal::check_vector(&rlwm, &truth, 1);

}

#[test]
fn rlwm_from_runs() {
    let width = 5;
    let runs = internal::random_integer_runs(51, width, 0.03);
    let truth = internal::runs_to_values(&runs);
    let rlwm = RLWM::from(runs);
    internal::check_vector(&rlwm, &truth, width);
}

#[test]
#[ignore]
fn large_rlwm() {
    let width = 10;
    let runs = internal::random_integer_runs(13842, width, 0.032);
    let truth = internal::runs_to_values(&runs);
    let rlwm = RLWM::from(runs);
    internal::check_vector(&rlwm, &truth, width);
}

//-----------------------------------------------------------------------------

// Serialize

#[test]
fn serialize_empty_rlwm() {
    let runs = Vec::new();
    let rlwm = RLWM::from(runs);
    serialize::test(&rlwm, "rl-wavelet-matrix", None, true);
}

#[test]
fn serialize_rlwm() {
    let runs = internal::random_integer_runs(42, 7, 0.08);
    let rlwm = RLWM::from(runs);
    serialize::test(&rlwm, "rl-wavelet-matrix", None, true);
}

#[test]
#[ignore]
fn serialize_large_rlwm() {
    let runs = internal::random_integer_runs(9836, 11, 0.025);
    let rlwm = RLWM::from(runs);
    serialize::test(&rlwm, "rl-wavelet-matrix", None, true);
}

//-----------------------------------------------------------------------------

// VectorIndex

#[test]
fn empty_rlwm_index() {
    let width = 1;
    let runs = Vec::new();
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

#[test]
fn rlwm_index() {
    let width = 4;
    let runs = internal::random_integer_runs(49, width, 0.04);
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

#[test]
fn rlwm_index_missing_values() {
    let width = 9;
    let runs = internal::random_integer_runs(38, width, 0.07);
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

#[test]
#[ignore]
fn large_rlwm_index() {
    let width = 6;
    let runs = internal::random_integer_runs(1011, width, 0.04);
    let rlwm = RLWM::from(runs);
    internal::check_vector_index(&rlwm, width);
}

//-----------------------------------------------------------------------------

// FIXME: Functionality specific to RLWM

//-----------------------------------------------------------------------------
