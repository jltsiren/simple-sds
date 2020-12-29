use super::*;

use crate::serialize;

use std::fs;

use rand::Rng;

//-----------------------------------------------------------------------------

#[test]
fn empty_vector() {
    let empty = RawVector::new();
    assert!(empty.is_empty(), "Created a non-empty empty vector");
    assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
    assert_eq!(empty.capacity(), 0, "Reserved unnecessary memory for an empty vector");

    let with_capacity = RawVector::with_capacity(137);
    assert!(with_capacity.is_empty(), "Created a non-empty vector by specifying capacity");
    assert!(with_capacity.capacity() >= 137, "Vector capacity is lower than specified");
}

#[test]
fn with_len_and_clear() {
    let mut v = RawVector::with_len(137, true);
    assert_eq!(v.len(), 137, "Vector length is not as specified");
    v.clear();
    assert!(v.is_empty(), "Could not clear the vector");
}

#[test]
fn initialization_vs_push() {
    let ones_set = RawVector::with_len(137, true);
    let mut ones_pushed = RawVector::new();
    for _ in 0..137 {
        ones_pushed.push_bit(true);
    }
    assert_eq!(ones_set, ones_pushed, "Initializing with and pushing ones yield different vectors");

    let zeros_set = RawVector::with_len(137, false);
    let mut zeros_pushed = RawVector::new();
    for _ in 0..137 {
        zeros_pushed.push_bit(false);
    }
    assert_eq!(zeros_set, zeros_pushed, "Initializing with and pushing zeros yield different vectors");
}

#[test]
fn initialization_vs_resize() {
    let initialized = RawVector::with_len(137, true);

    let mut extended = RawVector::with_len(66, true);
    extended.resize(137, true);
    assert_eq!(extended, initialized, "Extended vector is invalid");

    let mut truncated = RawVector::with_len(212, true);
    truncated.resize(137, true);
    assert_eq!(truncated, initialized, "Truncated vector is invalid");

    let mut popped = RawVector::with_len(97, true);
    for _ in 0..82 {
        popped.pop_bit();
    }
    popped.resize(137, true);
    assert_eq!(popped, initialized, "Popped vector is invalid after extension");
}

#[test]
fn reserving_capacity() {
    let mut original = RawVector::with_len(137, true);
    let copy = original.clone();
    original.reserve(31 + original.capacity() - original.len());

    assert!(original.capacity() >= 137 + 31, "Reserving additional capacity failed");
    assert_eq!(original, copy, "Reserving additional capacity changed the vector");
}

#[test]
fn push_bits() {
    let mut source: Vec<bool> = Vec::new();
    for i in 0..137 {
        source.push(i & 1 == 1);
    }

    let mut v = RawVector::new();
    for i in 0..137 {
        v.push_bit(source[i]);
    }
    assert_eq!(v.len(), 137, "Invalid vector length");

    let mut popped: Vec<bool> = Vec::new();
    while let Some(bit) = v.pop_bit() {
        popped.push(bit);
    }
    popped.reverse();
    assert!(v.is_empty(), "Non-empty vector after popping all bits");
    assert_eq!(popped, source, "Invalid sequence of bits popped from RawVector");
}

#[test]
fn set_bits() {
    let mut v = RawVector::with_len(137, false);
    let mut w = RawVector::with_len(137, true);
    assert_eq!(v.count_ones(), 0, "Non-zero bits in a zero-initialized vector");
    assert_eq!(w.count_ones(), 137, "Zero bits in an one-initialized vector");
    for i in 0..137 {
        v.set_bit(i, i & 1 == 1);
        w.set_bit(i, i & 1 == 1);
    }
    assert_eq!(v.len(), 137, "Invalid vector length");
    assert_eq!(v.count_ones(), 68, "Invalid number of ones in the vector");

    for i in 0..137 {
        assert_eq!(v.bit(i), i & 1 == 1, "Invalid bit {}", i);
    }
    assert_eq!(v, w, "Fully overwritten vector still depends on the initialization value");

    let complement = v.complement();
    for i in 0..137 {
        assert_eq!(!(complement.bit(i)), v.bit(i), "Invalid bit {} in the complement", i);
    }
}

#[test]
fn push_ints() {
    let mut correct: Vec<(u64, usize)> = Vec::new();
    for i in 0..64 {
        correct.push((i, 63)); correct.push((i * (i + 1), 64));
    }

    let mut v = RawVector::new();
    for (value, width) in correct.iter() {
        unsafe { v.push_int(*value, *width); }
    }
    assert_eq!(v.len(), 64 * (63 + 64), "Invalid vector length");

    correct.reverse();
    let mut popped: Vec<(u64, usize)> = Vec::new();
    for i in 0..correct.len() {
        let width = correct[i].1;
        if let Some(value) = unsafe { v.pop_int(width) } {
            popped.push((value, width));
        }
    }
    assert_eq!(popped.len(), correct.len(), "Invalid number of popped ints");
    assert!(v.is_empty(), "Non-empty vector after popping all ints");
    assert_eq!(popped, correct, "Invalid popped ints");
}

#[test]
fn set_ints() {
    let mut v = RawVector::with_len(64 * (63 + 64), false);
    let mut w = RawVector::with_len(64 * (63 + 64), true);
    let mut bit_offset = 0;
    for i in 0..64 {
        unsafe {
            v.set_int(bit_offset, i, 63); w.set_int(bit_offset, i, 63); bit_offset += 63;
            v.set_int(bit_offset, i * (i + 1), 64); w.set_int(bit_offset, i * (i + 1), 64); bit_offset += 64;
        }
    }
    assert_eq!(v.len(), 64 * (63 + 64), "Invalid vector length");

    bit_offset = 0;
    for i in 0..64 {
        unsafe {
            assert_eq!(v.int(bit_offset, 63), i, "Invalid integer [{}].0", i); bit_offset += 63;
            assert_eq!(v.int(bit_offset, 64), i * (i + 1), "Invalid integer [{}].1", i); bit_offset += 64;
        }
    }
    assert_eq!(v, w, "Fully overwritten vector still depends on the initialization value");
}

#[test]
fn get_words() {
    let correct: Vec<u64> = vec![0x123456, 0x789ABC, 0xFEDCBA, 0x987654];
    let mut v = RawVector::with_len(correct.len() * 64, false);
    for (index, value) in correct.iter().enumerate() {
        unsafe { v.set_int(index * 64, *value, 64); }
    }
    for (index, value) in correct.iter().enumerate() {
        assert_eq!(v.word(index), *value, "Invalid integer {}", index);
    }

    unsafe {
        for (index, value) in correct.iter().enumerate() {
            assert_eq!(v.word_unchecked(index), *value, "Invalid integer {} (unchecked)", index);
        }
    }
}

#[test]
fn serialize() {
    let mut original = RawVector::new();
    for i in 0..64 {
        unsafe { original.push_int(i * (i + 1) * (i + 2), 16); }
    }
    assert_eq!(original.size_in_bytes(), 144, "Invalid RawVector size in bytes");

    let filename = serialize::temp_file_name("raw-vector");
    serialize::serialize_to(&original, &filename).unwrap();

    let copy: RawVector = serialize::load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the RawVector");

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------

#[test]
fn empty_writer() {
    let first = serialize::temp_file_name("empty-raw-vector-writer");
    let second = serialize::temp_file_name("empty-raw-vector-writer");

    let mut v = RawVectorWriter::new(&first).unwrap();
    assert!(v.is_empty(), "Created a non-empty empty writer");
    assert_eq!(v.len(), 0, "Nonzero length for an empty writer");
    assert!(v.is_open(), "Newly created writer is not open");
    v.close().unwrap();

    let mut w = RawVectorWriter::with_buf_len(&second, 1024).unwrap();
    assert!(w.is_empty(), "Created a non-empty empty writer with custom buffer size");
    assert!(w.is_open(), "Newly created writer is not open with custom buffer size");
    w.close().unwrap();

    fs::remove_file(&first).unwrap();
    fs::remove_file(&second).unwrap();
}

#[test]
fn push_bits_to_writer() {
    let filename = serialize::temp_file_name("push-bits-to-raw-vector-writer");

    let mut correct: Vec<bool> = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..3523 {
        correct.push(rng.gen());
    }

    let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
    for bit in correct.iter() {
        v.push_bit(*bit);
    }
    assert_eq!(v.len(), correct.len(), "Invalid size for the writer after push_bit");
    v.close().unwrap();
    assert!(!v.is_open(), "Could not close the writer");

    let w: RawVector = serialize::load_from(&filename).unwrap();
    assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
    for i in 0..correct.len() {
        assert_eq!(w.bit(i), correct[i], "Invalid bit {}", i);
    }

    fs::remove_file(&filename).unwrap();
}

#[test]
fn push_ints_to_writer() {
    let filename = serialize::temp_file_name("push-ints-to-raw-vector-writer");

    let mut correct: Vec<u64> = Vec::new();
    let mut rng = rand::thread_rng();
    let width = 31;
    for _ in 0..71 {
        let value: u64 = rng.gen();
        correct.push(value & bits::low_set(width));
    }

    let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
    for value in correct.iter() {
        unsafe { v.push_int(*value, width); }
    }
    assert_eq!(v.len(), correct.len() * width, "Invalid size for the writer");
    v.close().unwrap();
    assert!(!v.is_open(), "Could not close the writer");

    let w: RawVector = serialize::load_from(&filename).unwrap();
    assert_eq!(w.len(), correct.len() * width, "Invalid size for the loaded vector");
    for i in 0..correct.len() {
        unsafe { assert_eq!(w.int(i * width, width), correct[i], "Invalid integer {}", i); }
    }

    fs::remove_file(&filename).unwrap();
}

#[test]
#[ignore]
fn large_writer() {
    let filename = serialize::temp_file_name("large-raw-vector-writer");

    let mut correct: Vec<u64> = Vec::new();
    let mut rng = rand::thread_rng();
    let width = 31;
    for _ in 0..620001 {
        let value: u64 = rng.gen();
        correct.push(value & bits::low_set(width));
    }

    let mut v = RawVectorWriter::new(&filename).unwrap();
    for value in correct.iter() {
        unsafe { v.push_int(*value, width); }
    }
    assert_eq!(v.len(), correct.len() * width, "Invalid size for the writer");
    v.close().unwrap();
    assert!(!v.is_open(), "Could not close the writer");

    let w: RawVector = serialize::load_from(&filename).unwrap();
    assert_eq!(w.len(), correct.len() * width, "Invalid size for the loaded vector");
    for i in 0..correct.len() {
        unsafe { assert_eq!(w.int(i * width, width), correct[i], "Invalid integer {}", i); }
    }

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------
