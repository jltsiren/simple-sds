//! Low-level functions for bit manipulation.
//!
//! Most operations are built around storing bits in integer arrays of type `u64` using the least significant bits first.

use std::ops::{Index, IndexMut};

//-----------------------------------------------------------------------------

/// Number of bytes in [`u64`].
pub const WORD_BYTES: usize = 8;

/// Number of bits in [`u64`].
pub const WORD_BITS: usize = 64;

// Bit shift for transforming a bit offset into an array index.
const INDEX_SHIFT: usize = 6;

// Bit mask for transforming a bit offset into an offset in [`u64`].
const OFFSET_MASK: usize = 0b111111;

//-----------------------------------------------------------------------------

const LOW_SET: [u64; 65] = [
    0x0000_0000_0000_0000,

    0x0000_0000_0000_0001, 0x0000_0000_0000_0003, 0x0000_0000_0000_0007, 0x0000_0000_0000_000F,
    0x0000_0000_0000_001F, 0x0000_0000_0000_003F, 0x0000_0000_0000_007F, 0x0000_0000_0000_00FF,
    0x0000_0000_0000_01FF, 0x0000_0000_0000_03FF, 0x0000_0000_0000_07FF, 0x0000_0000_0000_0FFF,
    0x0000_0000_0000_1FFF, 0x0000_0000_0000_3FFF, 0x0000_0000_0000_7FFF, 0x0000_0000_0000_FFFF,

    0x0000_0000_0001_FFFF, 0x0000_0000_0003_FFFF, 0x0000_0000_0007_FFFF, 0x0000_0000_000F_FFFF,
    0x0000_0000_001F_FFFF, 0x0000_0000_003F_FFFF, 0x0000_0000_007F_FFFF, 0x0000_0000_00FF_FFFF,
    0x0000_0000_01FF_FFFF, 0x0000_0000_03FF_FFFF, 0x0000_0000_07FF_FFFF, 0x0000_0000_0FFF_FFFF,
    0x0000_0000_1FFF_FFFF, 0x0000_0000_3FFF_FFFF, 0x0000_0000_7FFF_FFFF, 0x0000_0000_FFFF_FFFF,

    0x0000_0001_FFFF_FFFF, 0x0000_0003_FFFF_FFFF, 0x0000_0007_FFFF_FFFF, 0x0000_000F_FFFF_FFFF,
    0x0000_001F_FFFF_FFFF, 0x0000_003F_FFFF_FFFF, 0x0000_007F_FFFF_FFFF, 0x0000_00FF_FFFF_FFFF,
    0x0000_01FF_FFFF_FFFF, 0x0000_03FF_FFFF_FFFF, 0x0000_07FF_FFFF_FFFF, 0x0000_0FFF_FFFF_FFFF,
    0x0000_1FFF_FFFF_FFFF, 0x0000_3FFF_FFFF_FFFF, 0x0000_7FFF_FFFF_FFFF, 0x0000_FFFF_FFFF_FFFF,

    0x0001_FFFF_FFFF_FFFF, 0x0003_FFFF_FFFF_FFFF, 0x0007_FFFF_FFFF_FFFF, 0x000F_FFFF_FFFF_FFFF,
    0x001F_FFFF_FFFF_FFFF, 0x003F_FFFF_FFFF_FFFF, 0x007F_FFFF_FFFF_FFFF, 0x00FF_FFFF_FFFF_FFFF,
    0x01FF_FFFF_FFFF_FFFF, 0x03FF_FFFF_FFFF_FFFF, 0x07FF_FFFF_FFFF_FFFF, 0x0FFF_FFFF_FFFF_FFFF,
    0x1FFF_FFFF_FFFF_FFFF, 0x3FFF_FFFF_FFFF_FFFF, 0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
];

const HIGH_SET: [u64; 65] = [
    0x0000_0000_0000_0000,

    0x8000_0000_0000_0000, 0xC000_0000_0000_0000, 0xE000_0000_0000_0000, 0xF000_0000_0000_0000,
    0xF800_0000_0000_0000, 0xFC00_0000_0000_0000, 0xFE00_0000_0000_0000, 0xFF00_0000_0000_0000,
    0xFF80_0000_0000_0000, 0xFFC0_0000_0000_0000, 0xFFE0_0000_0000_0000, 0xFFF0_0000_0000_0000,
    0xFFF8_0000_0000_0000, 0xFFFC_0000_0000_0000, 0xFFFE_0000_0000_0000, 0xFFFF_0000_0000_0000,

    0xFFFF_8000_0000_0000, 0xFFFF_C000_0000_0000, 0xFFFF_E000_0000_0000, 0xFFFF_F000_0000_0000,
    0xFFFF_F800_0000_0000, 0xFFFF_FC00_0000_0000, 0xFFFF_FE00_0000_0000, 0xFFFF_FF00_0000_0000,
    0xFFFF_FF80_0000_0000, 0xFFFF_FFC0_0000_0000, 0xFFFF_FFE0_0000_0000, 0xFFFF_FFF0_0000_0000,
    0xFFFF_FFF8_0000_0000, 0xFFFF_FFFC_0000_0000, 0xFFFF_FFFE_0000_0000, 0xFFFF_FFFF_0000_0000,

    0xFFFF_FFFF_8000_0000, 0xFFFF_FFFF_C000_0000, 0xFFFF_FFFF_E000_0000, 0xFFFF_FFFF_F000_0000,
    0xFFFF_FFFF_F800_0000, 0xFFFF_FFFF_FC00_0000, 0xFFFF_FFFF_FE00_0000, 0xFFFF_FFFF_FF00_0000,
    0xFFFF_FFFF_FF80_0000, 0xFFFF_FFFF_FFC0_0000, 0xFFFF_FFFF_FFE0_0000, 0xFFFF_FFFF_FFF0_0000,
    0xFFFF_FFFF_FFF8_0000, 0xFFFF_FFFF_FFFC_0000, 0xFFFF_FFFF_FFFE_0000, 0xFFFF_FFFF_FFFF_0000,

    0xFFFF_FFFF_FFFF_8000, 0xFFFF_FFFF_FFFF_C000, 0xFFFF_FFFF_FFFF_E000, 0xFFFF_FFFF_FFFF_F000,
    0xFFFF_FFFF_FFFF_F800, 0xFFFF_FFFF_FFFF_FC00, 0xFFFF_FFFF_FFFF_FE00, 0xFFFF_FFFF_FFFF_FF00,
    0xFFFF_FFFF_FFFF_FF80, 0xFFFF_FFFF_FFFF_FFC0, 0xFFFF_FFFF_FFFF_FFE0, 0xFFFF_FFFF_FFFF_FFF0,
    0xFFFF_FFFF_FFFF_FFF8, 0xFFFF_FFFF_FFFF_FFFC, 0xFFFF_FFFF_FFFF_FFFE, 0xFFFF_FFFF_FFFF_FFFF,
];

//-----------------------------------------------------------------------------

// Lookup tables for the `select` implementation without the PDEP instruction.

// Each byte in `_PS_OVERFLOW[i]` contains `128 - i`.
const _PS_OVERFLOW: [u64; 65] = [
    0x8080_8080_8080_8080,

    0x7F7F_7F7F_7F7F_7F7F, 0x7E7E_7E7E_7E7E_7E7E, 0x7D7D_7D7D_7D7D_7D7D, 0x7C7C_7C7C_7C7C_7C7C,
    0x7B7B_7B7B_7B7B_7B7B, 0x7A7A_7A7A_7A7A_7A7A, 0x7979_7979_7979_7979, 0x7878_7878_7878_7878,
    0x7777_7777_7777_7777, 0x7676_7676_7676_7676, 0x7575_7575_7575_7575, 0x7474_7474_7474_7474,
    0x7373_7373_7373_7373, 0x7272_7272_7272_7272, 0x7171_7171_7171_7171, 0x7070_7070_7070_7070,

    0x6F6F_6F6F_6F6F_6F6F, 0x6E6E_6E6E_6E6E_6E6E, 0x6D6D_6D6D_6D6D_6D6D, 0x6C6C_6C6C_6C6C_6C6C,
    0x6B6B_6B6B_6B6B_6B6B, 0x6A6A_6A6A_6A6A_6A6A, 0x6969_6969_6969_6969, 0x6868_6868_6868_6868,
    0x6767_6767_6767_6767, 0x6666_6666_6666_6666, 0x6565_6565_6565_6565, 0x6464_6464_6464_6464,
    0x6363_6363_6363_6363, 0x6262_6262_6262_6262, 0x6161_6161_6161_6161, 0x6060_6060_6060_6060,

    0x5F5F_5F5F_5F5F_5F5F, 0x5E5E_5E5E_5E5E_5E5E, 0x5D5D_5D5D_5D5D_5D5D, 0x5C5C_5C5C_5C5C_5C5C,
    0x5B5B_5B5B_5B5B_5B5B, 0x5A5A_5A5A_5A5A_5A5A, 0x5959_5959_5959_5959, 0x5858_5858_5858_5858,
    0x5757_5757_5757_5757, 0x5656_5656_5656_5656, 0x5555_5555_5555_5555, 0x5454_5454_5454_5454,
    0x5353_5353_5353_5353, 0x5252_5252_5252_5252, 0x5151_5151_5151_5151, 0x5050_5050_5050_5050,

    0x4F4F_4F4F_4F4F_4F4F, 0x4E4E_4E4E_4E4E_4E4E, 0x4D4D_4D4D_4D4D_4D4D, 0x4C4C_4C4C_4C4C_4C4C,
    0x4B4B_4B4B_4B4B_4B4B, 0x4A4A_4A4A_4A4A_4A4A, 0x4949_4949_4949_4949, 0x4848_4848_4848_4848,
    0x4747_4747_4747_4747, 0x4646_4646_4646_4646, 0x4545_4545_4545_4545, 0x4444_4444_4444_4444,
    0x4343_4343_4343_4343, 0x4242_4242_4242_4242, 0x4141_4141_4141_4141, 0x4040_4040_4040_4040,
];

// `_SELECT_IN_BYTE[256 * i + x]` is the position of the `i`th set bit in `x`.
const _SELECT_IN_BYTE: [u8; 2048] = [
    0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
    4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,

    0, 0, 0, 1, 0, 2, 2, 1, 0, 3, 3, 1, 3, 2, 2, 1,
    0, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    0, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1,
    5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    0, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1,
    6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1,
    5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    0, 7, 7, 1, 7, 2, 2, 1, 7, 3, 3, 1, 3, 2, 2, 1,
    7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1,
    5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1,
    6, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
    6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1,
    5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,

    0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 3, 2,
    0, 0, 0, 4, 0, 4, 4, 2, 0, 4, 4, 3, 4, 3, 3, 2,
    0, 0, 0, 5, 0, 5, 5, 2, 0, 5, 5, 3, 5, 3, 3, 2,
    0, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    0, 0, 0, 6, 0, 6, 6, 2, 0, 6, 6, 3, 6, 3, 3, 2,
    0, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2,
    0, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2,
    6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    0, 0, 0, 7, 0, 7, 7, 2, 0, 7, 7, 3, 7, 3, 3, 2,
    0, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2,
    0, 7, 7, 5, 7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2,
    7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    0, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2,
    7, 6, 6, 4, 6, 4, 4, 2, 6, 4, 4, 3, 4, 3, 3, 2,
    7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2,
    6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
    0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 3,
    0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 3,
    0, 0, 0, 5, 0, 5, 5, 4, 0, 5, 5, 4, 5, 4, 4, 3,
    0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6, 3,
    0, 0, 0, 6, 0, 6, 6, 4, 0, 6, 6, 4, 6, 4, 4, 3,
    0, 0, 0, 6, 0, 6, 6, 5, 0, 6, 6, 5, 6, 5, 5, 3,
    0, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 3,
    0, 0, 0, 7, 0, 7, 7, 4, 0, 7, 7, 4, 7, 4, 4, 3,
    0, 0, 0, 7, 0, 7, 7, 5, 0, 7, 7, 5, 7, 5, 5, 3,
    0, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3,
    0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 3,
    0, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4, 6, 4, 4, 3,
    0, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3,
    7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
    0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 5, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
    0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6, 4,
    0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6, 5,
    0, 0, 0, 6, 0, 6, 6, 5, 0, 6, 6, 5, 6, 5, 5, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 4,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 5,
    0, 0, 0, 7, 0, 7, 7, 5, 0, 7, 7, 5, 7, 5, 5, 4,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6,
    0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 4,
    0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 5,
    0, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
    0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 6, 6, 5,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 5,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6,
    0, 0, 0, 7, 0, 7, 7, 6, 0, 7, 7, 6, 7, 6, 6, 5,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 6,

    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,
];

//-----------------------------------------------------------------------------

/// Returns an integer with the lowest `n` bits set.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::low_set(13), 0x1FFF);
/// ```
///
/// # Panics
///
/// May panic if `n > 64`.
#[inline]
pub fn low_set(n: usize) -> u64 {
    LOW_SET[n]
}

/// Unsafe version of [`low_set`].
///
/// # Safety
///
/// Behavior is undefined if `n > 64`.
#[inline]
pub unsafe fn low_set_unchecked(n: usize) -> u64 {
    *LOW_SET.get_unchecked(n)
}

/// Returns an integer with the highest `n` bits set.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::high_set(13), 0xFFF8_0000_0000_0000);
/// ```
///
/// # Panics
///
/// May panic if `n > 64`.
#[inline]
pub fn high_set(n: usize) -> u64 {
    HIGH_SET[n]
}

/// Unsafe version of [`high_set`].
///
/// # Safety
///
/// Behavior is undefined if `n > 64`.
#[inline]
pub unsafe fn high_set_unchecked(n: usize) -> u64 {
    *HIGH_SET.get_unchecked(n)
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
#[inline]
pub fn bit_len(n: u64) -> usize {
    // The `| 1` to avoid a branch was obvious in hindsight.
    WORD_BITS - ((n | 1).leading_zeros() as usize)
}

/// Returns the bit offset of the set bit of specified rank.
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
/// unsafe {
///     assert_eq!(bits::select(0b00100001_00010000, 0), 4);
///     assert_eq!(bits::select(0b00100001_00010000, 1), 8);
///     assert_eq!(bits::select(0b00100001_00010000, 2), 13);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if `rank >= n.count_ones()`.
#[inline]
pub unsafe fn select(n: u64, rank: usize) -> usize {
    // The first argument to `__pdep_u64` has a single 1 at bit offset `rank`. The
    // number `n` we are interested in is used as a mask. PDEP takes low-order bits
    // from the value and places them to the offsets specified by the mask. In
    // particular, the only 1 is placed to the bit offset of the set bit of rank
    // `rank` in `n`. We get the offset by counting the trailing zeros.
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        let pos = core::arch::x86_64::_pdep_u64(1u64 << rank, n);
        pos.trailing_zeros() as usize
    }

    // This is borrowed from SDSL.
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        // Each byte in `cumulative` will contain the cumulative number of set bits
        // in bytes up to and including that byte in `n`.
        let cumulative = n - ((n >> 1) & 0x5555_5555_5555_5555);
        let cumulative = (cumulative & 0x3333_3333_3333_3333) + ((cumulative >> 2) & 0x3333_3333_3333_3333);
        let cumulative = (cumulative + (cumulative >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        let (cumulative, _) = cumulative.overflowing_mul(0x0101_0101_0101_0101);

        // We add `128 - rank - 1` to each byte and mask out all bits except `128`. We get
        // the bit offset for the byte containing the answer by counting trailing zeros.
        let mask = (cumulative + *_PS_OVERFLOW.get_unchecked(rank + 1)) & 0x8080_8080_8080_8080;
        let offset = ((mask.trailing_zeros() >> 3) << 3) as usize;

        // Subtract the number of set bits in the previous bytes from the rank.
        let relative_rank = rank - (((cumulative << 8) >> offset) as usize & 0xFF);

        offset + (*_SELECT_IN_BYTE.get_unchecked((relative_rank << 8) + ((n >> offset) as usize & 0xFF)) as usize)
    }
}

//-----------------------------------------------------------------------------

/// Returns the number of bytes that can be stored in `n` integers of type [`u64`].
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::words_to_bytes(3), 24);
/// ```
///
/// # Panics
///
/// May panic if `n * 8 > usize::MAX`.
#[inline]
pub fn words_to_bytes(n: usize) -> usize {
    n * WORD_BYTES
}

/// Returns the number of integers of type [`u64`] required to store `n` bytes.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bytes_to_words(8), 1);
/// assert_eq!(bits::bytes_to_words(9), 2);
/// ```
///
/// # Panics
///
/// May panic if `n + 7 > usize::MAX`.
#[inline]
pub fn bytes_to_words(n: usize) -> usize {
    (n + WORD_BYTES - 1) / WORD_BYTES
}

/// Rounds `n` up to the next multiple of 8.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::round_up_to_word_bytes(0), 0);
/// assert_eq!(bits::round_up_to_word_bytes(8), 8);
/// assert_eq!(bits::round_up_to_word_bytes(9), 16);
/// ```
///
/// # Panics
///
/// May panic if `n + 7 > usize::MAX`.
#[inline]
pub fn round_up_to_word_bytes(n: usize) -> usize {
    words_to_bytes(bytes_to_words(n))
}

//-----------------------------------------------------------------------------

/// Returns the number of bits that can be stored in `n` integers of type [`u64`].
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::words_to_bits(3), 192);
/// ```
///
/// # Panics
///
/// May panic if `n * 64 > usize::MAX`.
#[inline]
pub fn words_to_bits(n: usize) -> usize {
    n * WORD_BITS
}

/// Returns the number of integers of type [`u64`] required to store `n` bits.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bits_to_words(64), 1);
/// assert_eq!(bits::bits_to_words(65), 2);
/// ```
///
/// # Panics
///
/// May panic if `n + 63 > usize::MAX`.
#[inline]
pub fn bits_to_words(n: usize) -> usize {
    (n + WORD_BITS - 1) / WORD_BITS
}

/// Rounds `n` up to the next multiple of 64.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::round_up_to_word_bits(0), 0);
/// assert_eq!(bits::round_up_to_word_bits(64), 64);
/// assert_eq!(bits::round_up_to_word_bits(65), 128);
/// ```
///
/// # Panics
///
/// May panic if `n + 63 > usize::MAX`.
#[inline]
pub fn round_up_to_word_bits(n: usize) -> usize {
    words_to_bits(bits_to_words(n))
}

//-----------------------------------------------------------------------------

/// Divides `value` by `n` and rounds the result up.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::div_round_up(129, 13), 10);
/// assert_eq!(bits::div_round_up(130, 13), 10);
/// assert_eq!(bits::div_round_up(131, 13), 11);
/// ```
///
/// # Panics
///
/// May panic if `value + n > usize::MAX` or `n == 0`.
#[inline]
pub fn div_round_up(value: usize, n: usize) -> usize {
    (value + n - 1) / n
}

/// Returns a [`u64`] value consisting entirely of bit `value`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::filler_value(false), 0);
/// assert_eq!(bits::filler_value(true), !0u64);
/// ```
#[inline]
pub fn filler_value(value: bool) -> u64 {
    match value {
        true  => !0u64,
        false => 0u64,
    }
}

/// Splits a bit offset into an index in an array of [`u64`] and an offset within the integer.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::split_offset(123), (1, 59));
/// ```
#[inline]
pub fn split_offset(bit_offset: usize) -> (usize, usize) {
    (bit_offset >> INDEX_SHIFT, bit_offset & OFFSET_MASK)
}

/// Combines an index in an array of [`u64`] and an offset within the integer into a bit offset.
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
///
/// # Panics
///
/// May panic if the result would be greater than [`usize::MAX`].
#[inline]
pub fn bit_offset(index: usize, offset: usize) -> usize {
    (index << INDEX_SHIFT) + offset
}

//-----------------------------------------------------------------------------

/// Writes an integer into a bit array implemented as an array of [`u64`] values.
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
/// unsafe {
///     bits::write_int(&mut array, 0, 7, 4);
///     bits::write_int(&mut array, 4, 3, 4);
/// }
/// assert_eq!(array[0], 0x37);
/// ```
///
/// # Safety
///
/// Behavior is undefined if `width > 64`.
///
/// # Panics
///
/// May panic if `(bit_offset + width - 1) / 64` is not a valid index in the array.
pub unsafe fn write_int<T: IndexMut<usize, Output = u64>>(array: &mut T, bit_offset: usize, value: u64, width: usize) {
    let value = value & low_set(width);
    let (index, offset) = split_offset(bit_offset);

    if offset + width <= WORD_BITS {
        array[index] &= high_set_unchecked(WORD_BITS - width - offset) | low_set_unchecked(offset);
        array[index] |= value << offset;
    } else {
        array[index] &= low_set_unchecked(offset);
        array[index] |= value << offset;
        array[index + 1] &= high_set(2 * WORD_BITS - width - offset);
        array[index + 1] |= value >> (WORD_BITS - offset);
    }
}

/// Reads an integer from a bit array implemented as an array of [`u64`] values.
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
/// unsafe {
///     assert_eq!(bits::read_int(&array, 0, 4), 7);
///     assert_eq!(bits::read_int(&array, 4, 4), 3);
/// }
/// ```
///
/// # Safety
///
/// Behavior is undefined if `width > 64`.
///
/// # Panics
///
/// May panic if `(bit_offset + width - 1) / 64` is not a valid index in the array.
pub unsafe fn read_int<T: Index<usize, Output = u64>>(array: &T, bit_offset: usize, width: usize) -> u64 {
    let (index, offset) = split_offset(bit_offset);
    let first = array[index] >> offset;

    if offset + width <= WORD_BITS {
        first & low_set_unchecked(width)
    } else {
        first | ((array[index + 1] & low_set_unchecked((offset + width) & OFFSET_MASK)) << (WORD_BITS - offset))
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
        assert_eq!(low_set(64), !0u64, "low_set(64) failed");

        unsafe {
            assert_eq!(low_set_unchecked(0), 0u64, "low_set_unchecked(0) failed");
            assert_eq!(low_set_unchecked(13), 0x1FFF, "low_set_unchecked(13) failed");
            assert_eq!(low_set_unchecked(64), !0u64, "low_set_unchecked(64) failed")
        }
    }

    #[test]
    fn high_set_test() {
        assert_eq!(high_set(0), 0, "high_set(0) failed");
        assert_eq!(high_set(13), 0xFFF8_0000_0000_0000, "high_set(13) failed");
        assert_eq!(high_set(64), !0u64, "high_set(64) failed");

        unsafe {
            assert_eq!(high_set_unchecked(0), 0, "high_set_unchecked(0) failed");
            assert_eq!(high_set_unchecked(13), 0xFFF8_0000_0000_0000, "high_set_unchecked(13) failed");
            assert_eq!(high_set_unchecked(64), !0u64, "high_set_unchecked(64) failed")
        }
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
    fn select_test() {
        let values: Vec<(u64, usize, usize)> = vec![
        (0x0000_0000_0000_0001, 0, 0),
        (0x8000_0000_0000_0000, 0, 63),
        (0x8000_0000_0000_0001, 0, 0),
        (0x8000_0000_0000_0001, 1, 63),
        (0x8000_0010_0000_0001, 0, 0),
        (0x8000_0010_0000_0001, 1, 36),
        (0x8000_0010_0000_0001, 2, 63),
        ];
        for (value, rank, result) in values.iter() {
            unsafe { assert_eq!(select(*value, *rank), *result, "select({:X}, {}) failed", value, rank); }
        }
    }

    #[test]
    fn read_write() {
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
            unsafe {
                write_int(&mut array, bit_offset, correct[i].0, 31); bit_offset += 31;
                write_int(&mut array, bit_offset, correct[i].1, 64); bit_offset += 64;
                write_int(&mut array, bit_offset, correct[i].2, 35); bit_offset += 35;
                write_int(&mut array, bit_offset, correct[i].3, 63); bit_offset += 63;
            }
        }

        bit_offset = 0;
        for i in 0..64 {
            unsafe {
                assert_eq!(read_int(&array, bit_offset, 31), correct[i].0, "Invalid value at [{}].0", i); bit_offset += 31;
                assert_eq!(read_int(&array, bit_offset, 64), correct[i].1, "Invalid value at [{}].1", i); bit_offset += 64;
                assert_eq!(read_int(&array, bit_offset, 35), correct[i].2, "Invalid value at [{}].2", i); bit_offset += 35;
                assert_eq!(read_int(&array, bit_offset, 63), correct[i].3, "Invalid value at [{}].3", i); bit_offset += 63;
            }
        }
    }

    #[test]
    fn no_extra_bits() {
        let mut array: Vec<u64> = vec![0; 2];
        unsafe {
            write_int(&mut array, 16, 2, 16);
            write_int(&mut array, 48, 2, 16);
            write_int(&mut array, 32, !0u64, 16); // This should not overwrite the integer at offset 48.
            write_int(&mut array, 0, !0u64, 16); // This should not overwrite the other integers.
            assert_eq!(read_int(&array, 0, 16), 0xFFFF, "Incorrect 16-bit integer at offset 0");
            assert_eq!(read_int(&array, 16, 16), 2, "Incorrect 16-bit integer at offset 16");
            assert_eq!(read_int(&array, 32, 16), 0xFFFF, "Incorrect 16-bit integer at offset 32");
            assert_eq!(read_int(&array, 48, 16), 2, "Incorrect 16-bit integer at offset 48");
        }
    }

    #[test]
    #[ignore]
    fn environment() {
        assert_eq!(mem::size_of::<usize>(), 8, "Things may not work if usize is not 64 bits");
        assert_eq!(mem::align_of_val(&1usize), 8, "Things may not work if the minimum alignment of usize is not 8 bytes");
        assert_eq!(mem::align_of_val(&1u64), 8, "Things may not work if the minimum alignment of u64 is not 8 bytes");
        assert!(cfg!(target_endian = "little"), "Things may not work on a big-endian system");
        assert!(cfg!(any(target_arch = "x86_64", target_arch = "aarch64")), "Target architecture should be x86_64 or aarch64");
        assert!(cfg!(unix), "Memory mapping requires Unix-like OS");

        #[cfg(target_arch = "x86_64")]
        {
            assert!(cfg!(target_feature = "popcnt"), "Performance may be worse without the POPCNT instruction");
            assert!(cfg!(target_feature = "lzcnt"), "Performance may be worse without the LZCNT instruction");
            assert!(cfg!(target_feature = "bmi1"), "Performance may be worse without the TZCNT instruction from BMI1");
            assert!(cfg!(target_feature = "bmi2"), "Performance may be worse without the PDEP instruction from BMI2");
        }
    }
}

//-----------------------------------------------------------------------------
