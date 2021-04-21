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
    assert_eq!(map.len(), correct.size_in_elements(), "Invalid file size for {}", name);

    let mapped = MappedSlice::<T>::new(&map, 0).unwrap();
    assert_eq!(mapped.is_empty(), correct.is_empty(), "Invaid emptiness for mapped slice of {}", name);
    assert_eq!(mapped.len(), correct.len(), "Invalid length for mapped slice of {}", name);
    for i in 0..mapped.len() {
        assert_eq!(mapped[i], correct[i], "Invalid value {} in {}", i, name);
    }
    assert_eq!(mapped.as_ref(), correct.as_slice(), "Invalid mapped slice for {}", name);
}

fn serialized_bytes<P: AsRef<Path>>(filename: P, correct: &Vec<u8>, name: &str) {
    assert_eq!(correct.size_in_bytes(), 8 + bits::round_up_to_word_bytes(correct.len()), "Invalid serialization size for {}", name);
    serialize_to(correct, &filename).unwrap();
    let copy: Vec<u8> = load_from(&filename).unwrap();
    assert_eq!(&copy, correct, "Serialization changed vector {}", name);
}

fn mapped_bytes<P: AsRef<Path>>(filename: P, correct: &Vec<u8>, name: &str) {
    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    assert!(!map.is_empty(), "The file is empty for vector {}", name);
    assert_eq!(map.len(), correct.size_in_elements(), "Invalid file size for {}", name);

    let mapped = MappedBytes::new(&map, 0).unwrap();
    assert_eq!(mapped.is_empty(), correct.is_empty(), "Invaid emptiness for mapped slice of {}", name);
    assert_eq!(mapped.len(), correct.len(), "Invalid length for mapped slice of {}", name);
    for i in 0..mapped.len() {
        assert_eq!(mapped[i], correct[i], "Invalid value {} in {}", i, name);
    }
    assert_eq!(mapped.as_ref(), correct.as_slice(), "Invalid mapped slice for {}", name);
}

fn serialized_string<P: AsRef<Path>>(filename: P, correct: &String, name: &str) {
    assert_eq!(correct.size_in_bytes(), 8 + bits::round_up_to_word_bytes(correct.len()), "Invalid serialization size for {}", name);
    serialize_to(correct, &filename).unwrap();
    let copy: String = load_from(&filename).unwrap();
    assert_eq!(&copy, correct, "Serialization changed string {}", name);
}

fn mapped_string<P: AsRef<Path>>(filename: P, correct: &String, name: &str) {
    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    assert!(!map.is_empty(), "The file is empty for string {}", name);
    assert_eq!(map.len(), correct.size_in_elements(), "Invalid file size for {}", name);

    let mapped = MappedStr::new(&map, 0).unwrap();
    assert_eq!(mapped.is_empty(), correct.is_empty(), "Invaid emptiness for mapped string slice of {}", name);
    assert_eq!(mapped.len(), correct.len(), "Invalid length for mapped string slice of {}", name);
    assert_eq!(mapped.as_ref(), correct.as_str(), "Invalid mapped string slice for {}", name);
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
    assert_eq!(map.len(), correct.size_in_elements(), "Invalid file size for {}", name);

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
fn serialize_bytes() {
    let filename = temp_file_name("vec-u8");

    let empty: Vec<u8> = Vec::new();
    serialized_bytes(&filename, &empty, "empty");
    mapped_bytes(&filename, &empty, "empty");

    let padded: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    serialized_bytes(&filename, &padded, "padded");
    mapped_bytes(&filename, &padded, "padded");

    let unpadded: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    serialized_bytes(&filename, &unpadded, "unpadded");
    mapped_bytes(&filename, &unpadded, "unpadded");

    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_string() {
    let filename = temp_file_name("string");

    let empty = String::new();
    serialized_string(&filename, &empty, "empty");
    mapped_string(&filename, &empty, "empty");

    let padded = String::from("0123456789ABC");
    serialized_string(&filename, &padded, "padded");
    mapped_string(&filename, &padded, "padded");

    let unpadded = String::from("0123456789ABCDEF");
    serialized_string(&filename, &unpadded, "unpadded");
    mapped_string(&filename, &unpadded, "unpadded");

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

#[test]
fn invalid_data() {
    let filename = temp_file_name("serialize-invalid-data");
    let mut options = OpenOptions::new();
    let mut file = options.create(true).write(true).truncate(true).open(&filename).unwrap();

    // The size of `u64` is one element, but the header indicates that it should take two.
    let size: usize = 2;
    let value: u64 = 123;
    size.serialize(&mut file).unwrap();
    value.serialize(&mut file).unwrap();
    drop(file);

    let result: io::Result<Option<u64>> = load_from(&filename);
    assert_eq!(result.map_err(|e| e.kind()), Err(ErrorKind::InvalidData), "Expected ErrorKind::InvalidData");

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------
