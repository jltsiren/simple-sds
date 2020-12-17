//! Low-level functions for bit manipulation.
//!
//! Most operations are built around storing bits in integer arrays of type `u64` using the least significant bits first.

use std::ops::{Index, IndexMut};

//-----------------------------------------------------------------------------

/// Number of bits in `u64`.
pub const WORD_BITS: usize = 64;

// Bit shift for transforming a bit offset into an array index.
const INDEX_SHIFT: usize = 6;

// Bit mask for transforming a bit offset into an offset in `u64`.
const OFFSET_MASK: usize = 0b111111;

//-----------------------------------------------------------------------------

/// Returns an integer with the lowest `n` bits set.
///
/// Behavior is undefined if `n > 64`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::low_set(13), 0x1FFF);
/// ```
pub fn low_set(n: usize) -> u64 {
    (0xFFFF_FFFF_FFFF_FFFFu128 >> (WORD_BITS - n)) as u64
}

/// Returns an integer with the highest `n` bits set.
///
/// Behavior is undefined if `n > 64`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::high_set(13), 0xFFF8_0000_0000_0000);
/// ```
pub fn high_set(n: usize) -> u64 {
    (0xFFFF_FFFF_FFFF_FFFFu128 << (WORD_BITS - n)) as u64
}

/// Returns the length of the binary representation of integer `n`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bit_len(0), 1);
/// assert_eq!(bits::bit_len(0x1FFF), 13);
/// ```
pub fn bit_len(n: u64) -> usize {
    match n {
        0 => 1,
        _ => WORD_BITS - (n.leading_zeros() as usize),
    }
}

// FIXME tests
/// Returns the bit offset of the set bit of specified rank.
///
/// Behavior is undefined if `rank >= n.count_ones()`.
///
/// # Arguments
///
/// * `n`: An integer.
/// * `rank`: Rank of the set bit we want to find.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::select(0b00100001_00010000, 0), 4);
/// assert_eq!(bits::select(0b00100001_00010000, 1), 8);
/// assert_eq!(bits::select(0b00100001_00010000, 2), 13);
/// ```
pub fn select(n: u64, rank: usize) -> usize {
    // The first argument to `__pdep_u64` has a single 1 at bit offset `rank`. The
    // number `n` we are interested in is used as a mask. PDEP takes low-order bits
    // from the value and places them to the offsets specified by the mask. In
    // particular, the only 1 is placed to the bit offset of the set bit of rank
    // `rank` in `n`. We get the offset by counting the trailing zeros.
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        let pos = unsafe { core::arch::x86_64::_pdep_u64(1u64 << rank, n) };
        pos.trailing_zeros() as usize
    }

    // TODO: A better implementation for ARM.

    // This is stupid and slow, but it's good enough fallback for now.
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut offset: usize = 0;
        let mut ones: usize = 0;
        loop {
            ones += ((n >> offset) & 1) as usize;
            if ones > rank {
                return offset;
            }
            offset += 1;
        }
    }
}

//-----------------------------------------------------------------------------

/// Returns the number of bits that can be stored in `n` integers of type `u64`.
///
/// Behavior is undefined if `n * 64 > usize::MAX`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::words_to_bits(3), 192);
/// ```
pub fn words_to_bits(n: usize) -> usize {
    n * WORD_BITS
}

/// Returns the number of integers of type `u64` required to store `n` bits.
///
/// Behavior is undefined if `n + 63 > usize::MAX`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bits_to_words(64), 1);
/// assert_eq!(bits::bits_to_words(65), 2);
/// ```
pub fn bits_to_words(n: usize) -> usize {
    (n + WORD_BITS - 1) / WORD_BITS
}

/// Rounds `n` up to the next positive multiple of 64.
///
/// Behavior is undefined if `n + 63 > usize::MAX`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::round_up_to_word_size(0), 64);
/// assert_eq!(bits::round_up_to_word_size(64), 64);
/// assert_eq!(bits::round_up_to_word_size(65), 128);
/// ```
pub fn round_up_to_word_size(n: usize) -> usize {
    match n {
        0 => WORD_BITS,
        _ => words_to_bits(bits_to_words(n)),
    }
}

/// Returns a `u64` value consisting entirely of bit `value`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::filler_value(false), 0);
/// assert_eq!(bits::filler_value(true), !0u64);
/// ```
pub fn filler_value(value: bool) -> u64 {
    match value {
        true  => !0u64,
        false => 0u64,
    }
}

/// Splits a bit offset into an index in an array of `u64` and an offset within the integer.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::split_offset(123), (1, 59));
/// ```
pub fn split_offset(bit_offset: usize) -> (usize, usize) {
    (bit_offset >> INDEX_SHIFT, bit_offset & OFFSET_MASK)
}

/// Combines an index in an array of `u64` and an offset within the integer into a bit offset.
///
/// # Arguments
///
/// * `index`: Array index.
/// * `offset`: Offset within the integer.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bit_offset(1, 59), 123);
/// ```
pub fn bit_offset(index: usize, offset: usize) -> usize {
    (index << INDEX_SHIFT) + offset
}

//-----------------------------------------------------------------------------

/// Writes an integer into a bit array implemented as an array of `u64` values.
///
/// Behavior is undefined if `width > 64`.
///
/// # Arguments
///
/// * `bit_offset`: Starting offset in the bit array.
/// * `value`: The integer to be written.
/// * `width`: The width of the integer in bits.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// let mut array: Vec<u64> = vec![0];
/// bits::write_int(&mut array, 0, 7, 4);
/// bits::write_int(&mut array, 4, 3, 4);
/// assert_eq!(array[0], 0x37);
/// ```
///
/// # Panics
///
/// Panics if `width > 64`.
/// May panic if `(bit_offset + width - 1) / 64` is not a valid index in the array.
pub fn write_int<T: IndexMut<usize, Output = u64>>(array: &mut T, bit_offset: usize, value: u64, width: usize) {

    let value = value & low_set(width);
    let (index, offset) = split_offset(bit_offset);

    if offset + width <= WORD_BITS {
        array[index] &= high_set(WORD_BITS - width - offset) | low_set(offset);
        array[index] |= value << offset;
    } else {
        array[index] &= low_set(offset);
        array[index] |= value << offset;
        array[index + 1] &= high_set(2 * WORD_BITS - width - offset);
        array[index + 1] |= value >> (WORD_BITS - offset);
    }
}

/// Reads an integer from an bit array implemented as an array of `u64` values.
///
/// Behavior is undefined if `width > 64`.
///
/// # Arguments
///
/// * `bit_offset`: Starting offset in the bit array.
/// * `width`: The width of the integer in bits.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// let array: Vec<u64> = vec![0x37];
/// assert_eq!(bits::read_int(&array, 0, 4), 7);
/// assert_eq!(bits::read_int(&array, 4, 4), 3);
/// ```
///
/// # Panics
///
/// May panic if `(bit_offset + width - 1) / 64` is not a valid index in the array.
pub fn read_int<T: Index<usize, Output = u64>>(array: &T, bit_offset: usize, width: usize) -> u64 {

    let (index, offset) = split_offset(bit_offset);
    let first = array[index] >> offset;

    if offset + width <= WORD_BITS {
        first & low_set(width)
    } else {
        first | ((array[index + 1] & low_set((offset + width) & OFFSET_MASK)) << (WORD_BITS - offset))
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    use rand::Rng;

    #[test]
    fn low_set_test() {
        assert_eq!(low_set(0), 0u64, "low_set(0) failed");
        assert_eq!(low_set(13), 0x1FFF, "low_set(13) failed");
        assert_eq!(low_set(64), !0u64, "low_set(64) failed")
    }

    #[test]
    fn high_set_test() {
        assert_eq!(high_set(0), 0, "high_set(0) failed");
        assert_eq!(high_set(13), 0xFFF8_0000_0000_0000, "high_set(13) failed");
        assert_eq!(high_set(64), !0u64, "high_set(64) failed")
    }

    #[test]
    fn bit_len_test() {
        assert_eq!(bit_len(0), 1, "bit_len(0) failed");
        assert_eq!(bit_len(1), 1, "bit_len(1) failed");
        assert_eq!(bit_len(0x1FFF), 13, "bit_len(0x1FFF) failed");
        assert_eq!(bit_len(0x7FFF_FFFF_FFFF_FFFF), 63, "bit_len(0x7FFF_FFFF_FFFF_FFFF) failed");
        assert_eq!(bit_len(0xFFFF_FFFF_FFFF_FFFF), 64, "bit_len(0xFFFF_FFFF_FFFF_FFFF) failed");
    }

    #[test]
    fn read_write_test() {
        let mut correct: Vec<(u64, u64, u64, u64)> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..64 {
            let mut tuple: (u64, u64, u64, u64) = rng.gen();
            tuple.0 &= low_set(31); tuple.1 &= low_set(64); tuple.2 &= low_set(35); tuple.3 &= low_set(63);
            correct.push(tuple);
        }

        let mut array: Vec<u64> = vec![0; 256];
        let mut bit_offset = 0;
        for i in 0..64 {
            write_int(&mut array, bit_offset, correct[i].0, 31); bit_offset += 31;
            write_int(&mut array, bit_offset, correct[i].1, 64); bit_offset += 64;
            write_int(&mut array, bit_offset, correct[i].2, 35); bit_offset += 35;
            write_int(&mut array, bit_offset, correct[i].3, 63); bit_offset += 63;
        }

        bit_offset = 0;
        for i in 0..64 {
            assert_eq!(read_int(&array, bit_offset, 31), correct[i].0, "Invalid value at [{}].0", i); bit_offset += 31;
            assert_eq!(read_int(&array, bit_offset, 64), correct[i].1, "Invalid value at [{}].1", i); bit_offset += 64;
            assert_eq!(read_int(&array, bit_offset, 35), correct[i].2, "Invalid value at [{}].2", i); bit_offset += 35;
            assert_eq!(read_int(&array, bit_offset, 63), correct[i].3, "Invalid value at [{}].3", i); bit_offset += 63;
        }
    }

    #[test]
    fn no_extra_bits() {
        let mut array: Vec<u64> = vec![0; 2];
        write_int(&mut array, 16, 2, 16);
        write_int(&mut array, 48, 2, 16);
        write_int(&mut array, 32, !0u64, 16); // This should not overwrite the integer at offset 48.
        write_int(&mut array, 0, !0u64, 16); // This should not overwrite the other integers.
        assert_eq!(read_int(&array, 0, 16), 0xFFFF, "Incorrect 16-bit integer at offset 0");
        assert_eq!(read_int(&array, 16, 16), 2, "Incorrect 16-bit integer at offset 16");
        assert_eq!(read_int(&array, 32, 16), 0xFFFF, "Incorrect 16-bit integer at offset 32");
        assert_eq!(read_int(&array, 48, 16), 2, "Incorrect 16-bit integer at offset 48");
    }

    #[test]
    #[ignore]
    fn environment() {
        assert_eq!(mem::size_of::<usize>(), 8, "Things may not work if usize is not 64 bits");
        assert!(cfg!(target_endian = "little"), "Things may not work on a big-endian system");
        assert!(cfg!(target_arch = "x86_64"), "Things may not work on architectures other than x86_64");

        assert!(cfg!(target_feature = "popcnt"), "Performance may be worse without the POPCNT instruction");
        assert!(cfg!(target_feature = "lzcnt"), "Performance may be worse without the LZCNT instruction");
        assert!(cfg!(target_feature = "bmi1"), "Performance may be worse without the TZCNT instruction from BMI1");
        assert!(cfg!(target_feature = "bmi2"), "Performance may be worse without the PDEP instruction from BMI2");
    }
}

//-----------------------------------------------------------------------------
