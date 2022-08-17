use super::*;

use crate::ops::{Vector, Resize, Pack, Access, Push, Pop};
use crate::serialize::{Serialize, MappingMode};
use crate::{serialize, internal};

use std::fs::OpenOptions;
use std::fs;

//-----------------------------------------------------------------------------

fn random_int_vector(n: usize, width: usize) -> IntVector {
    let values = internal::random_vector(n, width);
    let mut result = IntVector::new(width).unwrap();
    result.extend(values);
    result
}

//-----------------------------------------------------------------------------

#[test]
fn empty_vector() {
    let empty = IntVector::default();
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.width(), 64, "Invalid width for an empty vector");
    assert_eq!(empty.capacity(), 0, "Reserved unnecessary memory for an empty vector");

    let with_width = IntVector::new(13).unwrap();
    assert!(with_width.is_empty(), "Created a non-empty empty vector with a specified width");
    assert_eq!(with_width.width(), 13, "Invalid width for an empty vector with a specified width");
    assert_eq!(with_width.capacity(), 0, "Reserved unnecessary memory for an empty vector with a specified width");
    assert_eq!(with_width.max_len(), usize::MAX / 13, "Invalid maximum length");

    let with_capacity = IntVector::with_capacity(137, 13).unwrap();
    assert!(with_capacity.is_empty(), "Created a non-empty vector by specifying capacity");
    assert_eq!(with_width.width(), 13, "Invalid width for an empty vector with a specified capacity");
    assert!(with_capacity.capacity() >= 137, "Vector capacity is lower than specified");
    assert_eq!(with_capacity.max_len(), usize::MAX / 13, "Invalid maximum length");
}

#[test]
fn with_len_and_clear() {
    let mut v = IntVector::with_len(137, 13, 123).unwrap();
    assert_eq!(v.len(), 137, "Vector length is not as specified");
    assert_eq!(v.width(), 13, "Vector width is not as specified");
    v.clear();
    assert!(v.is_empty(), "Could not clear the vector");
    assert_eq!(v.width(), 13, "Clearing the vector changed its width");
}

#[test]
fn initialization_vs_push() {
    let with_len = IntVector::with_len(137, 13, 123).unwrap();
    let mut pushed = IntVector::new(13).unwrap();
    for _ in 0..137 {
        pushed.push(123);
    }
    assert_eq!(with_len, pushed, "Initializing with and pushing values yield different vectors");
}

#[test]
fn initialization_vs_resize() {
    let initialized = IntVector::with_len(137, 13, 123).unwrap();

    let mut extended = IntVector::with_len(66, 13, 123).unwrap();
    extended.resize(137, 123);
    assert_eq!(extended, initialized, "Extended vector is invalid");

    let mut truncated = IntVector::with_len(212, 13, 123).unwrap();
    truncated.resize(137, 123);
    assert_eq!(truncated, initialized, "Truncated vector is invalid");

    let mut popped = IntVector::with_len(97, 13, 123).unwrap();
    for _ in 0..82 {
        popped.pop();
    }
    popped.resize(137, 123);
    assert_eq!(popped, initialized, "Popped vector is invalid after extension");
}

#[test]
fn reserving_capacity() {
    let mut original = IntVector::with_len(137, 13, 123).unwrap();
    let copy = original.clone();
    original.reserve(31 + original.capacity() - original.len());

    assert!(original.capacity() >= 137 + 31, "Reserving additional capacity failed");
    assert_eq!(original, copy, "Reserving additional capacity changed the vector");
}

#[test]
fn push_pop_from_iter() {
    let mut correct: Vec<u16> = Vec::new();
    for i in 0..64 {
        correct.push(i); correct.push(i * (i + 1));
    }

    let mut v = IntVector::new(16).unwrap();
    for value in correct.iter() {
        v.push(*value as u64);
    }
    assert_eq!(v.len(), 128, "Invalid vector length");

    let from_iter: IntVector = correct.iter().cloned().collect();
    assert_eq!(from_iter, v, "Vector built from an iterator is invalid");

    correct.reverse();
    let mut popped: Vec<u16> = Vec::new();
    for _ in 0..correct.len() {
        if let Some(value) = v.pop() {
            popped.push(value as u16);
        }
    }
    assert_eq!(popped.len(), correct.len(), "Invalid number of popped ints");
    assert!(v.is_empty(), "Non-empty vector after popping all ints");
    assert_eq!(popped, correct, "Invalid popped ints");
}

#[test]
fn set_get() {
    let mut v = IntVector::with_len(128, 13, 0).unwrap();
    for i in 0..64 {
        v.set(2 * i, i as u64); v.set(2 * i + 1, (i * (i + 1)) as u64);
    }
    for i in 0..64 {
        assert_eq!(v.get(2 * i), i as u64, "Invalid integer [{}].0", i);
        assert_eq!(v.get(2 * i + 1), (i * (i + 1)) as u64, "Invalid integer [{}].1", i);
    }

    let raw = RawVector::from(v.clone());
    assert_eq!(raw.len(), v.len() * v.width(), "Invalid length for the extracted RawVector");
    for i in 0..v.len() {
        unsafe { assert_eq!(raw.int(i * v.width(), v.width()), v.get(i), "Invalid value {} in the RawVector", i); }
    }
}

#[test]
fn from_vec() {
    let correct: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let int_vec = IntVector::from(correct.clone());
    internal::check_vector(&int_vec, &correct, 64);
}

#[test]
fn extend() {
    let first: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let second: Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128];
    let mut correct: Vec<u64> = Vec::new();
    correct.extend(first.iter().cloned()); correct.extend(second.iter().cloned());

    let mut int_vec = IntVector::new(8).unwrap();
    int_vec.extend(first); int_vec.extend(second);
    internal::check_vector(&int_vec, &correct, 8);
}

#[test]
fn pack() {
    let correct: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];

    let mut packed: IntVector = correct.iter().cloned().collect();
    packed.pack();
    internal::check_vector(&packed, &correct, 7);
}

#[test]
fn serialize() {
    let mut original = IntVector::new(16).unwrap();
    for i in 0..64 {
        original.push(i * (i + 1) * (i + 2));
    }
    let _ = serialize::test(&original, "int-vector", Some(20), true);
}

#[test]
fn invalid_data() {
    let filename = serialize::temp_file_name("int-vector-invalid-data");
    let mut options = OpenOptions::new();
    let mut file = options.create(true).write(true).truncate(true).open(&filename).unwrap();

    // 13 values of 20 bits each will take 260 bits, but data length is 240 bits
    let len: usize = 13;
    let width: usize = 20;
    let data = RawVector::with_len(240, false);
    len.serialize(&mut file).unwrap();
    width.serialize(&mut file).unwrap();
    data.serialize(&mut file).unwrap();
    drop(file);

    let result: io::Result<IntVector> = serialize::load_from(&filename);
    assert_eq!(result.map_err(|e| e.kind()), Err(ErrorKind::InvalidData), "Expected ErrorKind::InvalidData");

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------

#[test]
fn empty_writer() {
    let first = serialize::temp_file_name("empty-int-vector-writer");
    let second = serialize::temp_file_name("empty-int-vector-writer");

    let mut v = IntVectorWriter::new(&first, 13).unwrap();
    assert!(v.is_empty(), "Created a non-empty empty writer");
    assert!(v.is_open(), "Newly created writer is not open");
    assert_eq!(v.filename(), first, "Invalid file name");
    v.close().unwrap();

    let mut w = IntVectorWriter::with_buf_len(&second, 13, 1024).unwrap();
    assert!(w.is_empty(), "Created a non-empty empty writer with custom buffer size");
    assert!(w.is_open(), "Newly created writer is not open with custom buffer size");
    assert_eq!(w.filename(), second, "Invalid file name with custom buffer size");
    w.close().unwrap();

    fs::remove_file(&first).unwrap();
    fs::remove_file(&second).unwrap();
}

#[test]
fn push_to_writer() {
    let filename = serialize::temp_file_name("push-to-int-vector-writer");

    let width = 31;
    let correct = internal::random_vector(71, width);

    let mut v = IntVectorWriter::with_buf_len(&filename, width, 32).unwrap();
    for value in correct.iter() {
        v.push(*value);
    }
    assert_eq!(v.len(), correct.len(), "Invalid size for the writer");
    v.close().unwrap();
    assert!(!v.is_open(), "Could not close the writer");

    let w: IntVector = serialize::load_from(&filename).unwrap();
    assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
    for i in 0..correct.len() {
        assert_eq!(w.get(i), correct[i], "Invalid integer {}", i);
    }

    fs::remove_file(&filename).unwrap();
}

#[test]
fn extend_writer() {
    let filename = serialize::temp_file_name("extend-int-vector-writer");

    let first: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let second: Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128];
    let mut correct: Vec<u64> = Vec::new();
    correct.extend(first.iter().cloned()); correct.extend(second.iter().cloned());

    let mut writer = IntVectorWriter::with_buf_len(&filename, 8, 32).unwrap();
    writer.extend(first); writer.extend(second);
    assert_eq!(writer.len(), correct.len(), "Invalid length for the extended writer");
    writer.close().unwrap();

    let int_vec: IntVector = serialize::load_from(&filename).unwrap();
    let collected: Vec<u64> = int_vec.into_iter().collect();
    assert_eq!(collected, correct, "Invalid values collected from the writer");

    fs::remove_file(&filename).unwrap();
}

#[test]
#[ignore]
fn large_writer() {
    let filename = serialize::temp_file_name("large-int-vector-writer");

    let width = 31;
    let correct: Vec<u64> = internal::random_vector(620001, width);

    let mut v = IntVectorWriter::new(&filename, width).unwrap();
    for value in correct.iter() {
        v.push(*value);
    }
    assert_eq!(v.len(), correct.len(), "Invalid size for the writer");
    v.close().unwrap();
    assert!(!v.is_open(), "Could not close the writer");

    let w: IntVector = serialize::load_from(&filename).unwrap();
    assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
    for i in 0..correct.len() {
        assert_eq!(w.get(i), correct[i], "Invalid integer {}", i);
    }

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------

fn check_mapper(mapper: &IntVectorMapper, truth: &IntVector) {
    assert!(!mapper.is_mutable(), "Read-only mapper is mutable");
    assert_eq!(mapper.len(), truth.len(), "Invalid mapper length");
    assert_eq!(mapper.is_empty(), truth.is_empty(), "Invalid mapper emptiness");
    assert_eq!(mapper.width(), truth.width(), "Invalid mapper width");

    for i in 0..mapper.len() {
        assert_eq!(mapper.get(i), truth.get(i), "Invalid value {}", i);
    }
    assert!(mapper.iter().eq(truth.iter()), "Invalid iterator (forward)");

    let mut index = mapper.len();
    let mut iter = mapper.iter();
    while let Some(value) = iter.next_back() {
        index -= 1;
        assert_eq!(value, truth.get(index), "Invalid value {} when iterating backwards", index);
    }
}

#[test]
fn empty_mapper() {
    let filename = serialize::temp_file_name("empty-int-vector-mapper");
    let truth = IntVector::new(17).unwrap();
    serialize::serialize_to(&truth, &filename).unwrap();

    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    let mapper = IntVectorMapper::new(&map, 0).unwrap();
    check_mapper(&mapper, &truth);

    drop(mapper); drop(map);
    fs::remove_file(&filename).unwrap();
}

#[test]
fn non_empty_mapper() {
    let filename = serialize::temp_file_name("non-empty-int-vector-mapper");
    let truth = random_int_vector(318, 29);
    serialize::serialize_to(&truth, &filename).unwrap();

    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    let mapper = IntVectorMapper::new(&map, 0).unwrap();
    check_mapper(&mapper, &truth);

    drop(mapper); drop(map);
    fs::remove_file(&filename).unwrap();
}

#[test]
fn mapper_offset() {
    let filename = serialize::temp_file_name("int-vector-mapper-offset");
    let width: usize = 33;
    let truth = random_int_vector(251, width);

    let mut options = OpenOptions::new();
    let mut file = options.create(true).write(true).truncate(true).open(&filename).unwrap();
    width.serialize(&mut file).unwrap(); // One element of padding.
    truth.serialize(&mut file).unwrap();
    drop(file);

    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    let mapper = IntVectorMapper::new(&map, 1).unwrap();
    check_mapper(&mapper, &truth);

    drop(mapper); drop(map);
    fs::remove_file(&filename).unwrap();
}

#[test]
#[ignore]
fn large_mapper() {
    let filename = serialize::temp_file_name("large-int-vector-mapper");
    let truth = random_int_vector(813752, 37);
    serialize::serialize_to(&truth, &filename).unwrap();

    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    let mapper = IntVectorMapper::new(&map, 0).unwrap();
    check_mapper(&mapper, &truth);

    drop(mapper); drop(map);
    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------
