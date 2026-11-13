//! Positioned file I/O and host memory reporting (no external crates).

use std::fs::File;
use std::io;

#[cfg(windows)]
use std::os::windows::fs::FileExt;
#[cfg(unix)]
use std::os::unix::fs::FileExt;

pub fn pread(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(windows)]
        let n = f.seek_read(&mut buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let n = f.read_at(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read at offset {off}"),
            ));
        }
        done += n;
    }
    Ok(())
}

pub fn pwrite(f: &File, buf: &[u8], off: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(windows)]
        let n = f.seek_write(&buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let n = f.write_at(&buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "write returned 0"));
        }
        done += n;
    }
    Ok(())
}

#[cfg(windows)]
pub fn free_memory_bytes() -> u64 {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: the struct is initialized and the size field is set as the API requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        8 << 30
    } else {
        status.avail_phys
    }
}

#[cfg(not(windows))]
pub fn free_memory_bytes() -> u64 {
    8 << 30
}
