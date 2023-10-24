use super::*;

use crate::internal;

//-----------------------------------------------------------------------------

// A naive integer vector for testing the default implementations in traits.
struct NaiveVector(Vec<u64>);

impl From<Vec<u64>> for NaiveVector {
    fn from(source: Vec<u64>) -> Self {
        NaiveVector(source)
    }
}

impl Vector for NaiveVector {
    type Item = u64;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn width(&self) -> usize {
        64
    }

    fn max_len(&self) -> usize {
        usize::MAX
    }
}

impl<'a> Access<'a> for NaiveVector {
    type Iter = AccessIter<'a, Self>;

    fn get(&self, index: usize) -> <Self as Vector>::Item {
        self.0[index]
    }

    fn iter(&'a self) -> Self::Iter {
        Self::Iter::new(self)
    }
}

struct ValueIter<'a> {
    parent: &'a NaiveVector,
    value: u64,
    rank: usize,
    index: usize,
}

impl<'a> Iterator for ValueIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.parent.len() {
            if self.parent.get(self.index) == self.value {
                let result = Some((self.rank, self.index));
                self.rank += 1; self.index += 1;
                return result;
            }
            self.index += 1;
        }
        None
    }
}

impl<'a> VectorIndex<'a> for NaiveVector {
    type ValueIter = ValueIter<'a>;

    fn rank(&self, index: usize, value: <Self as Vector>::Item) -> usize {
        let index = cmp::min(index, self.len());
        let mut result = 0;
        for i in 0..index {
            if self.get(i) == value {
                result += 1;
            }
        }
        result
    }

    fn value_iter(&'a self, value: <Self as Vector>::Item) -> Self::ValueIter {
        Self::ValueIter {
            parent: self,
            value,
            rank: 0,
            index: 0,
        }
    }

    fn value_of(iter: &Self::ValueIter) -> <Self as Vector>::Item {
        iter.value
    }

    fn select(&self, rank: usize, value: <Self as Vector>::Item) -> Option<usize> {
        let mut found = 0;
        for index in 0..self.len() {
            if self.get(index) == value {
                if found == rank {
                    return Some(index);
                }
                found += 1;
            }
        }
        None
    }

    fn select_iter(&'a self, rank: usize, value: <Self as Vector>::Item) -> Self::ValueIter {
        let index = self.select(rank, value).unwrap_or(self.len());
        Self::ValueIter { parent: self, value, rank, index, }
    }
}

//-----------------------------------------------------------------------------

#[test]
fn get_or() {
    let data: Vec<u64> = (0..237).collect();
    let naive = NaiveVector::from(data);
    let default = u64::MAX;

    for i in 0..naive.len() {
        assert_eq!(naive.get_or(i, default), i as u64, "Invalid get_or({})", i);
    }
    assert_eq!(naive.get_or(naive.len(), default), default, "Invalid default value from get_or()");
}

#[test]
fn access_iter() {
    let data = internal::random_vector(322, 7);
    let naive = NaiveVector::from(data.clone());

    assert!(naive.iter().eq(data.iter().cloned()), "Invalid values from iterator");

    let mut naive_iter = naive.iter();
    let mut data_iter = data.iter().cloned();
    while let Some(value) = naive_iter.next_back() {
        assert_eq!(Some(value), data_iter.next_back(), "Invalid values from reverse iterator");
    }
    assert!(data_iter.next_back().is_none(), "Did not get enough values from reverse iterator");
}

#[test]
fn access_iter_nth() {
    let data = internal::random_vector(271, 6);
    let naive = NaiveVector::from(data);

    // Forward.
    for i in 0..naive.len() {
        assert_eq!(naive.iter().nth(i), Some(naive.get(i)), "Invalid nth({})", i);
    }

    // Backward.
    for i in 0..naive.len() {
        assert_eq!(naive.iter().nth_back(i), Some(naive.get(naive.len() - 1 - i)), "Invalid nth_back({})", i);
    }
}

#[test]
fn contains() {
    let width = 8;
    let data = internal::random_vector(198, width);
    let naive = NaiveVector::from(data);
    internal::check_contains(&naive, width);
}

#[test]
fn inverse_select() {
    let width = 6;
    let data = internal::random_vector(322, width);
    let naive = NaiveVector::from(data);
    internal::check_inverse_select(&naive);
}

#[test]
fn pred_succ() {
    let width = 7;
    let data = internal::random_vector(179, width);
    let naive = NaiveVector::from(data);
    internal::check_pred_succ(&naive, width);
}

//-----------------------------------------------------------------------------

