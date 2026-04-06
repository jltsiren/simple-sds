//! Utility functions for binaries.

use std::path::Path;

//-----------------------------------------------------------------------------

const UNITS: [(f64, &str); 6] = [
    (1.0, "B"),
    (1024.0, "KiB"),
    (1024.0 * 1024.0, "MiB"),
    (1024.0 * 1024.0 * 1024.0, "GiB"),
    (1024.0 * 1024.0 * 1024.0 * 1024.0, "TiB"),
    (1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0, "PiB"),
];

/// Returns a human-readable representation of a size in bytes.
///
/// # Examples
///
/// ```
/// use simple_sds::binaries;
///
/// let (size, unit) = binaries::human_readable_size(2560);
/// assert_eq!(size, 2.5);
/// assert_eq!(unit, "KiB");
///
/// let (size, unit) = binaries::human_readable_size(3 * 1024 * 1024 * 1024);
/// assert_eq!(size, 3.0);
/// assert_eq!(unit, "GiB");
/// ```
pub fn human_readable_size(bytes: usize) -> (f64, &'static str) {
    let value = bytes as f64;
    let mut unit = UNITS[0];
    for next in UNITS.iter().skip(1) {
        if value >= next.0 {
            unit = *next;
        } else {
            break;
        }
    }

    (value / unit.0, unit.1)
}

const SUFFIXES: [(&str, usize); 10] = [
    ("k", 1_000),
    ("m", 1_000_000),
    ("g", 1_000_000_000),
    ("t", 1_000_000_000_000),
    ("p", 1_000_000_000_000_000),
    ("ki", 1024),
    ("mi", 1024 * 1024),
    ("gi", 1024 * 1024 * 1024),
    ("ti", 1024 * 1024 * 1024 * 1024),
    ("pi", 1024 * 1024 * 1024 * 1024 * 1024),
];

/// Parses a human-readable representation of an unsigned integer.
///
/// The input string may have a magnitude suffix, possibly separated by whitespace.
/// The suffixes are case-insensitive and interpreted as metric prefixes or size units.
///
/// # Examples
///
/// ```
/// use simple_sds::binaries;
///
/// assert_eq!(binaries::parse_unsigned("2.5 KiB").unwrap(), 2560);
/// assert_eq!(binaries::parse_unsigned("3GiB").unwrap(), 3 * 1024 * 1024 * 1024);
/// assert_eq!(binaries::parse_unsigned("1.5M").unwrap(), 1_500_000);
/// assert_eq!(binaries::parse_unsigned("500k").unwrap(), 500_000);
/// assert_eq!(binaries::parse_unsigned("500Ki").unwrap(), 512_000);
/// assert_eq!(binaries::parse_unsigned("1024").unwrap(), 1024);
/// assert!(binaries::parse_unsigned("invalid").is_err());
/// ```
pub fn parse_unsigned(value: &str) -> Result<usize, String> {
    let value = value.trim();
    let mut numeric = String::new();
    let mut iter = value.chars().peekable();
    let mut float = false;
    while let Some(&c) = iter.peek() {
        if c.is_ascii_digit() || c == '.' {
            if c == '.' {
                float = true;
            }
            numeric.push(c);
            iter.next();
        } else {
            break;
        }
    }
    while let Some(&c) = iter.peek() {
        if c.is_whitespace() {
            iter.next();
        } else {
            break;
        }
    }
    let suffix: String = iter.map(|c| c.to_ascii_lowercase()).collect();

    let multiplier = {
        let m = if suffix.ends_with('b') {
            &suffix[..suffix.len() - 1]
        } else {
            &suffix
        };
        if m.is_empty() {
            1
        } else {
            SUFFIXES.iter().find(|(s, _)| *s == m).map(|(_, v)| *v).ok_or_else(||
                format!("Cannot parse numeric value: \"{}\"", value)
            )?
        }
    };

    if float {
        let float_value: f64 = numeric.parse().map_err(|_| format!("Cannot parse numeric value: \"{}\"", value))?;
        Ok((float_value * (multiplier as f64)).round() as usize)
    } else {
        let int_value: usize = numeric.parse().map_err(|_| format!("Cannot parse numeric value: \"{}\"", value))?;
        Ok(int_value * multiplier)
    }
}

//-----------------------------------------------------------------------------

/// Returns the peak resident set size (RSS) in bytes.
///
/// This function is only available on Linux and macOS when the `libc` feature is enabled.
///
/// # Errors
///
/// Returns an error if the `getrusage` system call fails or if there is no implementation for the current OS.
#[cfg(all(target_os = "linux", feature = "libc"))]
pub fn peak_memory_usage() -> Result<usize, String> {
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        let retval = libc::getrusage(libc::RUSAGE_SELF, &mut rusage as *mut _);
        match retval {
            0 => Ok(rusage.ru_maxrss as usize * 1024),
            val => Err(format!("libc::getrusage call failed with return value {}", val)),
        }
    }
}

/// Returns the peak resident set size (RSS) in bytes.
///
/// This function is only available on Linux and macOS when the `libc` feature is enabled.
///
/// # Errors
///
/// Returns an error if the `getrusage` system call fails or if there is no implementation for the current OS.
#[cfg(all(target_os = "macos", feature = "libc"))]
pub fn peak_memory_usage() -> Result<usize, String> {
    unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        let retval = libc::getrusage(libc::RUSAGE_SELF, &mut rusage as *mut _);
        match retval {
            0 => Ok(rusage.ru_maxrss as usize),
            val => Err(format!("libc::getrusage call failed with return value {}", val)),
        }
    }
}

/// Returns the peak resident set size (RSS) in bytes.
///
/// This function is only available on Linux and macOS when the `libc` feature is enabled.
///
/// # Errors
///
/// Returns an error if the `getrusage` system call fails or if there is no implementation for the current OS.
#[cfg(not(all(any(target_os = "linux", target_os = "macos"), feature = "libc")))]
pub fn peak_memory_usage() -> Result<usize, String> {
    Err(String::from("No peak_memory_usage implementation for this OS"))
}

//-----------------------------------------------------------------------------

/// Returns `true` if the file exists.
pub fn file_exists<P: AsRef<Path>>(filename: P) -> bool {
    std::fs::metadata(filename).is_ok()
}

/// Returns a human-readable representation of the file size in bytes.
///
/// Returns [`None`] if the file does not exist or the size cannot be determined.
/// See [`human_readable_size`] for the formatting of the size.
pub fn file_size<P: AsRef<Path>>(filename: P) -> Option<(f64, &'static str)> {
    let metadata = std::fs::metadata(filename);
    if metadata.is_err() {
        return None;
    }
    Some(human_readable_size(metadata.unwrap().len() as usize))
}

//-----------------------------------------------------------------------------
