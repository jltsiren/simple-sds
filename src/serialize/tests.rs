use super::*;
use std::fmt::Debug;
use std::{fs, mem};

//-----------------------------------------------------------------------------

fn serialized_vector<P, T>(filename: P, correct: &Vec<T>, name: &str) where
    P: AsRef<Path>,
    T: Serializable + Debug + PartialEq
{
    assert_eq!(correct.size_in_bytes(), 8 + mem::size_of::<T>() * correct.len(), "Invalid serialization size for {}", name);
    serialize_to(correct, &filename).unwrap();
    let copy: Vec<T> = load_from(&filename).unwrap();
    assert_eq!(&copy, correct, "Serialization changed vector {}", name);
}

fn mapped_vector<P, T>(filename: P, correct: &Vec<T>, name: &str) where
    P: AsRef<Path>,
    T: Serializable + Debug + PartialEq
{
    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    assert!(!map.is_empty(), "The file is empty for vector {}", name);
    assert_eq!(map.len(), correct.size_in_bytes() / 8, "Invalid file size for {}", name);

    let mapped = MappedSlice::<T>::new(&map, 0).unwrap();
    assert_eq!(mapped.is_empty(), correct.is_empty(), "Invaid emptiness for mapped slice of {}", name);
    assert_eq!(mapped.len(), correct.len(), "Invalid length for mapped slice of {}", name);
    for i in 0..mapped.len() {
        assert_eq!(mapped[i], correct[i], "Invalid value {} in {}", i, name);
    }
    assert_eq!(mapped.as_ref(), correct.as_slice(), "Invalid mapped slice for {}", name);
}

fn serialized_option<P: AsRef<Path>>(filename: P, correct: &Option<Vec<u64>>, name: &str) {
    let expected_size: usize = 8 + match correct {
        Some(value) => value.size_in_bytes(),
        None => 0,
    };
    assert_eq!(correct.size_in_bytes(), expected_size, "Invalid serialization size for {}", name);
    serialize_to(correct, &filename).unwrap();
    let copy: Option<Vec<u64>> = load_from(&filename).unwrap();
    assert_eq!(&copy, correct, "Serialization changed option {}", name);
}

fn mapped_option<P: AsRef<Path>>(filename: P, correct: &Option<Vec<u64>>, name: &str) {
    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    assert!(!map.is_empty(), "The file is empty for option {}", name);
    assert_eq!(map.len(), correct.size_in_bytes() / 8, "Invalid file size for {}", name);

    let mapped = MappedOption::<MappedSlice<u64>>::new(&map, 0).unwrap();
    assert_eq!(mapped.is_some(), correct.is_some(), "Invalid is_some() for {}", name);
    assert_eq!(mapped.is_none(), correct.is_none(), "Invalid is_none() for {}", name);
    if mapped.is_some() {
        assert_eq!(mapped.unwrap(), mapped.as_ref().unwrap(), "Different results with unwrap() options for {}", name);
        assert_eq!(mapped.unwrap().as_ref(), correct.as_ref().unwrap().as_slice(), "Invalid content for {}", name);
    } else {
        assert_eq!(mapped.as_ref(), None, "Got a Some value for {}", name);
    }
}

//-----------------------------------------------------------------------------

#[test]
fn serialize_usize() {
    let filename = temp_file_name("usize");

    let original: usize = 0x1234_5678_9ABC_DEF0;
    assert_eq!(original.size_in_bytes(), 8, "Invalid serialized size for usize");
    serialize_to(&original, &filename).unwrap();

    let copy: usize = load_from(&filename).unwrap();
    assert_eq!(copy, original, "Serialization changed the value of usize");

    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_vec_u64() {
    let filename = temp_file_name("vec-u64");

    let empty: Vec<u64> = Vec::new();
    serialized_vector(&filename, &empty, "empty");
    mapped_vector(&filename, &empty, "empty");

    let original: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    serialized_vector(&filename, &original, "non-empty");
    mapped_vector(&filename, &original, "non-empty");

    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_vec_u64_u64() {
    let filename = temp_file_name("vec-u64-u64");

    let empty: Vec<(u64, u64)> = Vec::new();
    serialized_vector(&filename, &empty, "empty");
    mapped_vector(&filename, &empty, "empty");

    let original: Vec<(u64, u64)> = vec![(1, 1), (2, 3), (5, 8), (13, 21), (34, 55), (89, 144)];
    serialized_vector(&filename, &original, "non-empty");
    mapped_vector(&filename, &original, "non-empty");

    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_option() {
    let filename = temp_file_name("option");

    let none: Option<Vec<u64>> = None;
    serialized_option(&filename, &none, "none");
    mapped_option(&filename, &none, "none");

    let some: Option<Vec<u64>> = Some(vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]);
    serialized_option(&filename, &some, "some");
    mapped_option(&filename, &some, "some");

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------
