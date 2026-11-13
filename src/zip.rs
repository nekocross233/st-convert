//! Tiny zip reader: only what a torch `.pth` archive needs (stored entries,
//! zip64 for >4 GiB archives).

use crate::sysio::pread;
use std::fs::File;
use std::io;

pub struct Entry {
    pub name: String,
    pub method: u16,
    pub size: u64,
    pub local_off: u64,
}

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const EOCD64_LOC_SIG: u32 = 0x0706_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

/// Read the central directory. Returns entries in archive order.
pub fn read_dir(f: &File, file_len: u64) -> io::Result<Vec<Entry>> {
    if file_len < 22 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for zip"));
    }
    let tail_len = std::cmp::min(66 * 1024, file_len) as usize;
    let tail_off = file_len - tail_len as u64;
    let mut tail = vec![0u8; tail_len];
    pread(f, &mut tail, tail_off)?;

    let mut eocd = None;
    for i in (0..=tail_len.saturating_sub(22)).rev() {
        if u32_at(&tail, i) == EOCD_SIG {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "zip EOCD not found"))?;
    let eocd_abs = tail_off + eocd as u64;

    let mut entries_total = u16_at(&tail, eocd + 10) as u64;
    let mut cd_size = u32_at(&tail, eocd + 12) as u64;
    let mut cd_off = u32_at(&tail, eocd + 16) as u64;

    if entries_total == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        // zip64: locator sits right before the EOCD
        if eocd_abs < 20 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad zip64 locator"));
        }
        let mut loc = [0u8; 20];
        pread(f, &mut loc, eocd_abs - 20)?;
        if u32_at(&loc, 0) != EOCD64_LOC_SIG {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zip64 locator missing"));
        }
        let eocd64_off = u64_at(&loc, 8);
        let mut rec = [0u8; 56];
        pread(f, &mut rec, eocd64_off)?;
        if u32_at(&rec, 0) != EOCD64_SIG {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zip64 EOCD missing"));
        }
        entries_total = u64_at(&rec, 32);
        cd_size = u64_at(&rec, 40);
        cd_off = u64_at(&rec, 48);
    }

    let mut cd = vec![0u8; cd_size as usize];
    pread(f, &mut cd, cd_off)?;

    let mut entries = Vec::with_capacity(entries_total as usize);
    let mut p = 0usize;
    while p + 46 <= cd.len() {
        if u32_at(&cd, p) != CENTRAL_SIG {
            break;
        }
        let method = u16_at(&cd, p + 10);
        let mut size = u32_at(&cd, p + 24) as u64;
        let name_len = u16_at(&cd, p + 28) as usize;
        let extra_len = u16_at(&cd, p + 30) as usize;
        let comment_len = u16_at(&cd, p + 32) as usize;
        let mut local_off = u32_at(&cd, p + 42) as u64;
        let name =
            String::from_utf8_lossy(&cd[p + 46..p + 46 + name_len]).into_owned();

        // zip64 extended information extra field (0x0001)
        let extra_off = p + 46 + name_len;
        let mut e = 0usize;
        while e + 4 <= extra_len {
            let id = u16_at(&cd, extra_off + e);
            let sz = u16_at(&cd, extra_off + e + 2) as usize;
            if id == 0x0001 {
                let mut q = extra_off + e + 4;
                if size == 0xFFFF_FFFF && q + 8 <= extra_off + e + 4 + sz {
                    size = u64_at(&cd, q);
                    q += 8;
                }
                if local_off == 0xFFFF_FFFF && q + 8 <= extra_off + e + 4 + sz {
                    local_off = u64_at(&cd, q);
                }
            }
            e += 4 + sz;
        }

        entries.push(Entry {
            name,
            method,
            size,
            local_off,
        });
        p = extra_off + extra_len + comment_len;
    }

    Ok(entries)
}

/// Byte offset of the entry payload (after the local header).
pub fn payload_offset(f: &File, entry: &Entry) -> io::Result<u64> {
    let mut h = [0u8; 30];
    pread(f, &mut h, entry.local_off)?;
    if u32_at(&h, 0) != LOCAL_SIG {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad local header"));
    }
    let name_len = u16_at(&h, 26) as u64;
    let extra_len = u16_at(&h, 28) as u64;
    Ok(entry.local_off + 30 + name_len + extra_len)
}

/// Read a whole (stored) entry.
pub fn read_entry(f: &File, entry: &Entry) -> io::Result<Vec<u8>> {
    if entry.method != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("zip entry {} is compressed (method {}); only stored entries are supported", entry.name, entry.method),
        ));
    }
    let off = payload_offset(f, entry)?;
    let mut buf = vec![0u8; entry.size as usize];
    pread(f, &mut buf, off)?;
    Ok(buf)
}
