use super::*;
use std::fmt::Debug;
use std::fs;

//-----------------------------------------------------------------------------

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
    assert_eq!(*mapped, *correct, "Invalid mapped slice for {}", name);
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
    assert_eq!(*mapped, *correct, "Invalid mapped slice for {}", name);
}

fn mapped_string<P: AsRef<Path>>(filename: P, correct: &String, name: &str) {
    let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    assert!(!map.is_empty(), "The file is empty for string {}", name);
    assert_eq!(map.len(), correct.size_in_elements(), "Invalid file size for {}", name);

    let mapped = MappedStr::new(&map, 0).unwrap();
    assert_eq!(mapped.is_empty(), correct.is_empty(), "Invaid emptiness for mapped string slice of {}", name);
    assert_eq!(mapped.len(), correct.len(), "Invalid length for mapped string slice of {}", name);
    assert_eq!(*mapped, *correct, "Invalid mapped string slice for {}", name);
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
    let original: usize = 0x1234_5678_9ABC_DEF0;
    let _ = test(&original, "usize", Some(1), true);
}

#[test]
fn serialize_vec_u64() {
    let empty: Vec<u64> = Vec::new();
    let filename = test(&empty, "empty-vec-u64", Some(1), false).unwrap();
    mapped_vector(&filename, &empty, "empty-vec-u64");
    fs::remove_file(&filename).unwrap();

    let original: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let filename = test(&original, "non-empty-vec-u64", Some(1 + original.len()), false).unwrap();
    mapped_vector(&filename, &original, "non-empty-vec-u64");
    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_vec_u64_u64() {
    let empty: Vec<(u64, u64)> = Vec::new();
    let filename = test(&empty, "empty-vec-u64-u64", Some(1), false).unwrap();
    mapped_vector(&filename, &empty, "empty-vec-u64-u64");
    fs::remove_file(&filename).unwrap();

    let original: Vec<(u64, u64)> = vec![(1, 1), (2, 3), (5, 8), (13, 21), (34, 55), (89, 144)];
    let filename = test(&original, "non-empty-vec-u64-u64", Some(1 + 2 * original.len()), false).unwrap();
    mapped_vector(&filename, &original, "non-empty-vec-u64-u64");
    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_bytes() {
    let empty: Vec<u8> = Vec::new();
    let filename = test(&empty, "empty-vec-u8", Some(1), false).unwrap();
    mapped_bytes(&filename, &empty, "empty-vec-u8");
    fs::remove_file(&filename).unwrap();

    let padded: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let filename = test(&padded, "padded-vec-u8", Some(1 + bits::bytes_to_words(padded.len())), false).unwrap();
    mapped_bytes(&filename, &padded, "padded-vec-u8");
    fs::remove_file(&filename).unwrap();

    let unpadded: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let filename = test(&unpadded, "unpadded-vec-u8", Some(1 + bits::bytes_to_words(unpadded.len())), false).unwrap();
    mapped_bytes(&filename, &unpadded, "unpadded-vec-u8");
    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_string() {
    let empty = String::new();
    let filename = test(&empty, "empty-string", Some(1), false).unwrap();
    mapped_string(&filename, &empty, "empty-string");
    fs::remove_file(&filename).unwrap();

    let padded = String::from("0123456789ABC");
    let filename = test(&padded, "padded-string", Some(1 + bits::bytes_to_words(padded.len())), false).unwrap();
    mapped_string(&filename, &padded, "padded-string");
    fs::remove_file(&filename).unwrap();

    let unpadded = String::from("0123456789ABCDEF");
    let filename = test(&unpadded, "unpadded-string", Some(1 + bits::bytes_to_words(unpadded.len())), false).unwrap();
    mapped_string(&filename, &unpadded, "unpadded-string");
    fs::remove_file(&filename).unwrap();
}

#[test]
fn serialize_option() {
    let none: Option<Vec<u64>> = None;
    let filename = test(&none, "none-option", Some(1), false).unwrap();
    mapped_option(&filename, &none, "none-option");
    fs::remove_file(&filename).unwrap();

    let some: Option<Vec<u64>> = Some(vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]);
    let filename = test(&some, "some-option", Some(1 + 1 + some.as_ref().unwrap().len()), false).unwrap();
    mapped_option(&filename, &some, "some-option");
    fs::remove_file(&filename).unwrap();
}

#[test]
fn skip_options() {
    let none: Option<usize> = None;
    let first: usize = 123;
    let some: Option<usize> = Some(456);
    let second: usize = 789;
    let third: usize = 101112;

    let filename = temp_file_name("skip-option");
    let mut options = OpenOptions::new();
    let mut file = options.create(true).write(true).truncate(true).open(&filename).unwrap();
    none.serialize(&mut file).unwrap();
    first.serialize(&mut file).unwrap();
    some.serialize(&mut file).unwrap();
    second.serialize(&mut file).unwrap();
    absent_option(&mut file).unwrap();
    third.serialize(&mut file).unwrap();
    drop(file);

    let mut options = OpenOptions::new();
    let mut file = options.read(true).open(&filename).unwrap();
    skip_option(&mut file).unwrap();
    assert_eq!(usize::load(&mut file).unwrap(), first, "Invalid value after skipped empty option");
    skip_option(&mut file).unwrap();
    assert_eq!(usize::load(&mut file).unwrap(), second, "Invalid value after skipped non-empty option");
    assert_eq!(Option::<usize>::load(&mut file).unwrap(), None, "Invalid absent option");
    assert_eq!(usize::load(&mut file).unwrap(), third, "Invalid value after absent option");
    drop(file);

    fs::remove_file(&filename).unwrap();
}

//-----------------------------------------------------------------------------
