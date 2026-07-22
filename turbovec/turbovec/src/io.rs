//! Read/write TurboVec index files.
//!
//! Two formats live here:
//! * `.tv` — [`TurboQuantIndex`](crate::TurboQuantIndex) — 4-byte magic
//!   "TVPI" + version + bit_width/dim/n_vectors header + packed codes +
//!   per-vector scales + (v3+) TQ+ per-coord calibration.
//! * `.tvim` — [`IdMapIndex`](crate::IdMapIndex) — 4-byte magic "TVIM"
//!   + version + the same core-index payload + a trailing `slot_to_id`
//!   table of `u64` values.
//!
//! ## Format versioning
//!
//! Both formats are at version 3 as of turbovec 0.6.x (TQ+ per-coord
//! calibration). Version 2 (turbovec 0.4.4 .. 0.6.0) is loaded transparently
//! with empty calibration — the index behaves like the old encoding, with
//! no recall change and no TQ+ gain. Re-encoding from source vectors picks
//! up the new calibration. Version 1 (turbovec ≤ 0.4.3) is incompatible
//! and refused with a rebuild hint.
//!
//! Version 1 `.tv` files had no magic — the file started with a bare
//! bit_width byte (2/3/4). Version 2+ prepends magic + version, which
//! lets us detect either a current file or "looks like a v1 turbovec
//! file" cleanly.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TV_VERSION: u8 = 3;
const TVPB_MAGIC: &[u8; 4] = b"TVPB";
const TVPB_VERSION: u8 = 1;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION: u8 = 3;

const REBUILD_HINT: &str =
    "Rebuild this index from the source vectors using turbovec 0.4.4 or later \
     (no in-place migration is provided; the format version 2 changes the meaning \
     of the per-vector scalar from ||v|| to a length-renormalization correction).";

/// Core payload — what a fully-deserialized index needs.
pub(crate) type CoreLoad = (usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Blocked-layout payload used by the query-major benchmark. The blocked
/// bytes are already in the architecture-specific layout consumed by the
/// SIMD scorer, so a request can load a range without repacking it first.
pub(crate) type BlockedLoad = (
    usize,
    usize,
    usize,
    usize,
    Vec<u8>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
);

/// `.tv` write — positional index.
pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TV_MAGIC)?;
    f.write_all(&[TV_VERSION])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;
    f.flush()?;
    Ok(())
}

/// Write a query-optimised, SIMD-blocked sidecar. The sidecar deliberately
/// uses a separate magic/version so the canonical packed `.tv` format remains
/// backwards compatible. Its layout tag prevents an x86 blocked file from
/// being accidentally consumed by the ARM sequential scorer (or vice versa).
pub(crate) fn write_blocked(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    blocked_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let n_byte_groups = dim / (8 / bit_width);
    let expected_blocked = n_blocks
        .checked_mul(n_byte_groups)
        .and_then(|x| x.checked_mul(crate::BLOCK))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "blocked size overflows usize"))?;
    if blocked_codes.len() != expected_blocked || scales.len() != n_vectors {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "blocked payload shape mismatch: codes={}, expected {}; scales={}, expected {}",
                blocked_codes.len(), expected_blocked, scales.len(), n_vectors
            ),
        ));
    }
    if tqplus_shift.len() != tqplus_scale.len()
        || (!tqplus_shift.is_empty() && tqplus_shift.len() != dim)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid TQ+ calibration lengths",
        ));
    }

    f.write_all(TVPB_MAGIC)?;
    f.write_all(&[TVPB_VERSION, blocked_layout_tag(), bit_width as u8])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;
    f.write_all(&(n_blocks as u32).to_le_bytes())?;
    f.write_all(blocked_codes)?;
    for &s in scales {
        f.write_all(&s.to_le_bytes())?;
    }
    let n_calib = tqplus_shift.len() as u32;
    f.write_all(&n_calib.to_le_bytes())?;
    for &s in tqplus_shift {
        f.write_all(&s.to_le_bytes())?;
    }
    for &s in tqplus_scale {
        f.write_all(&s.to_le_bytes())?;
    }
    f.flush()
}

/// `.tv` load — positional index. Transparently handles v2 (no TQ+) and
/// v3 (with TQ+) files; v2 returns empty TQ+ vectors which the engine
/// treats as identity calibration.
pub fn load(path: impl AsRef<Path>) -> io::Result<CoreLoad> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV_MAGIC {
        // Version 1 .tv files had no magic — first byte was the bit_width
        // (always 2, 3, or 4). If we see one of those as the first byte,
        // emit a targeted error rather than the generic "wrong magic"
        // message; otherwise treat it as a non-turbovec file.
        if (2..=4).contains(&magic[0]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "this .tv file was written by turbovec ≤ 0.4.3 (format \
                     version 1). It is incompatible with turbovec 0.4.4+ \
                     because the per-vector scalar's meaning changed. {}",
                    REBUILD_HINT,
                ),
            ));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tv file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    read_core_versioned(&mut f, version[0], TV_VERSION, ".tv")
}

/// Load one positional range from a `.tv` file without materialising the
/// complete compressed index. The returned index has local slots numbered
/// from zero; callers that scan multiple ranges must add the range offset to
/// returned ids.
pub(crate) fn load_range(
    path: impl AsRef<Path>,
    start_vector: usize,
    count: usize,
) -> io::Result<CoreLoad> {
    let mut f = BufReader::new(File::open(path)?);
    load_range_reader(&mut f, start_vector, count)
}

/// Load one positional range from an already-open reader. Keeping the reader
/// open is useful to bounded-memory callers that scan many ranges per
/// request: the range seeks remain positional, but file-open/header setup is
/// not repeated for every range.
pub(crate) fn load_range_reader<R: Read + Seek>(
    mut f: &mut R,
    start_vector: usize,
    count: usize,
) -> io::Result<CoreLoad> {
    // A caller may reuse the reader for multiple ranges. Header offsets are
    // relative to the beginning of the file, so reset before parsing each
    // range rather than assuming the reader is newly opened.
    f.seek(SeekFrom::Start(0))?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tv file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if !matches!(version[0], 2 | 3) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported .tv format version: {} (this build supports versions 2 and {})",
                version[0], TV_VERSION
            ),
        ));
    }

    let (bit_width, dim, n_vectors) = read_validated_header(&mut f)?;
    if dim == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cannot load a range from an uncommitted lazy index",
        ));
    }
    if start_vector > n_vectors || count > n_vectors - start_vector {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "range [{start_vector}, {}) is outside index with {n_vectors} vectors",
                start_vector.saturating_add(count)
            ),
        ));
    }

    let header_end = f.stream_position()?;
    let bytes_per_vector = (dim / 8)
        .checked_mul(bit_width)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed code size overflows usize"))?;
    let total_code_bytes = bytes_per_vector
        .checked_mul(n_vectors)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed code size overflows usize"))?;
    let range_code_bytes = bytes_per_vector
        .checked_mul(count)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "range code size overflows usize"))?;

    let code_offset = header_end
        .checked_add(
            bytes_per_vector
                .checked_mul(start_vector)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "range offset overflows file size"))?
                as u64,
        )
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "range offset overflows file size"))?;
    f.seek(SeekFrom::Start(code_offset))?;
    let packed_codes = read_exact_vec(&mut f, range_code_bytes)?;

    let scales_offset = header_end
        .checked_add(total_code_bytes as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "scale offset overflows file size"))?;
    let range_scales_offset = scales_offset
        .checked_add(
            start_vector
                .checked_mul(4)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "scale offset overflows file size"))?
                as u64,
        )
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "scale offset overflows file size"))?;
    f.seek(SeekFrom::Start(range_scales_offset))?;
    let scales = read_f32_array(&mut f, count)?;

    let calibration_offset = scales_offset
        .checked_add(
            n_vectors
                .checked_mul(4)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "calibration offset overflows file size"))?
                as u64,
        )
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "calibration offset overflows file size"))?;
    let (tqplus_shift, tqplus_scale) = if version[0] == 3 {
        f.seek(SeekFrom::Start(calibration_offset))?;
        read_tqplus_trailer(&mut f, dim)?
    } else {
        (Vec::new(), Vec::new())
    };

    Ok((
        bit_width,
        dim,
        count,
        packed_codes,
        scales,
        tqplus_shift,
        tqplus_scale,
    ))
}

/// Load one range from a persisted blocked sidecar. `start_vector` must be
/// block-aligned; the final range may be shorter than one full block.
pub(crate) fn load_blocked_range_reader<R: Read + Seek>(
    mut f: &mut R,
    start_vector: usize,
    count: usize,
) -> io::Result<BlockedLoad> {
    f.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVPB_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec blocked sidecar: wrong magic",
        ));
    }
    let mut fixed = [0u8; 3];
    f.read_exact(&mut fixed)?;
    if fixed[0] != TVPB_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported blocked sidecar version: {}", fixed[0]),
        ));
    }
    if fixed[1] != blocked_layout_tag() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "blocked sidecar layout does not match this CPU architecture",
        ));
    }
    let bit_width = fixed[2] as usize;
    if !matches!(bit_width, 2 | 3 | 4 | 8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid blocked bit_width {bit_width}"),
        ));
    }
    let mut nums = [0u8; 12];
    f.read_exact(&mut nums)?;
    let dim = u32::from_le_bytes(nums[0..4].try_into().unwrap()) as usize;
    let n_vectors = u32::from_le_bytes(nums[4..8].try_into().unwrap()) as usize;
    let n_blocks = u32::from_le_bytes(nums[8..12].try_into().unwrap()) as usize;
    if dim == 0 || dim % 8 != 0 || dim > crate::MAX_DIM {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid blocked dim {dim}"),
        ));
    }
    let expected_blocks = (n_vectors + crate::BLOCK - 1) / crate::BLOCK;
    if n_blocks != expected_blocks || start_vector > n_vectors || count > n_vectors - start_vector {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid blocked sidecar vector range",
        ));
    }
    if start_vector % crate::BLOCK != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "blocked sidecar range must start on a 32-vector boundary",
        ));
    }
    let end_vector = start_vector + count;
    let block_count = (count + crate::BLOCK - 1) / crate::BLOCK;
    if end_vector < n_vectors && count % crate::BLOCK != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "non-final blocked sidecar range must contain whole blocks",
        ));
    }

    let n_byte_groups = dim / (8 / bit_width);
    let bytes_per_block = n_byte_groups
        .checked_mul(crate::BLOCK)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked block size overflows usize"))?;
    let total_blocked_bytes = n_blocks
        .checked_mul(bytes_per_block)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked payload size overflows usize"))?;
    let header_end = f.stream_position()?;
    let range_offset = header_end
        .checked_add(
            (start_vector / crate::BLOCK)
                .checked_mul(bytes_per_block)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked range offset overflows"))?
                as u64,
        )
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked range offset overflows"))?;
    f.seek(SeekFrom::Start(range_offset))?;
    let blocked_codes = read_exact_vec(
        &mut f,
        block_count
            .checked_mul(bytes_per_block)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked range size overflows"))?,
    )?;

    let scales_offset = header_end
        .checked_add(total_blocked_bytes as u64)
        .and_then(|x| x.checked_add((start_vector * 4) as u64))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked scales offset overflows"))?;
    f.seek(SeekFrom::Start(scales_offset))?;
    let scales = read_f32_array(&mut f, count)?;

    let calibration_offset = header_end
        .checked_add(total_blocked_bytes as u64)
        .and_then(|x| x.checked_add((n_vectors * 4) as u64))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blocked calibration offset overflows"))?;
    f.seek(SeekFrom::Start(calibration_offset))?;
    let (tqplus_shift, tqplus_scale) = read_tqplus_trailer(&mut f, dim)?;

    Ok((
        bit_width,
        dim,
        count,
        block_count,
        blocked_codes,
        scales,
        tqplus_shift,
        tqplus_scale,
    ))
}

#[cfg(target_arch = "aarch64")]
const fn blocked_layout_tag() -> u8 {
    1
}

#[cfg(target_arch = "x86_64")]
const fn blocked_layout_tag() -> u8 {
    2
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const fn blocked_layout_tag() -> u8 {
    3
}

/// `.tvim` write — positional index plus the id-map side-tables.
pub fn write_id_map(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u64],
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );

    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVIM_MAGIC)?;
    f.write_all(&[TVIM_VERSION])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;

    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// `.tvim` load — positional index plus the id-map side-tables.
pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>)> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVIM_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVIM file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if version[0] == 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "this .tvim file was written by turbovec ≤ 0.4.3 (format \
                 version 1). It is incompatible with turbovec 0.4.4+ \
                 because the per-vector scalar's meaning changed. {}",
                REBUILD_HINT,
            ),
        ));
    }
    let (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale) =
        read_core_versioned(&mut f, version[0], TVIM_VERSION, ".tvim")?;

    // Read the slot_to_id table via the capped reader rather than
    // `Vec::with_capacity(n_vectors)` — `n_vectors` is attacker-controlled and
    // pre-reserving it allows a tiny file to drive a huge allocation.
    let id_bytes = n_vectors
        .checked_mul(8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "id table size overflows usize"))?;
    let raw = read_exact_vec(&mut f, id_bytes)?;
    let slot_to_id: Vec<u64> = raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();

    Ok((
        bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale,
        slot_to_id,
    ))
}

const CORE_HEADER_SIZE: usize = 9;

/// Core header + packed codes + per-vector scales + TQ+ calibration —
/// shared by `.tv` and `.tvim`.
fn write_core<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    w.write_all(&[bit_width as u8])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u32).to_le_bytes())?;
    w.write_all(packed_codes)?;
    for &s in scales {
        w.write_all(&s.to_le_bytes())?;
    }
    // TQ+ trailer. n_calib == 0 means identity calibration (lazy index
    // with no add yet, or a loaded pre-TQ+ index that's been resaved);
    // otherwise must equal dim.
    assert!(
        tqplus_shift.len() == tqplus_scale.len()
            && (tqplus_shift.is_empty() || tqplus_shift.len() == dim),
        "TQ+ shift/scale must have equal length and either be empty or equal dim"
    );
    let n_calib = tqplus_shift.len() as u32;
    w.write_all(&n_calib.to_le_bytes())?;
    for &s in tqplus_shift {
        w.write_all(&s.to_le_bytes())?;
    }
    for &s in tqplus_scale {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Read the core payload, dispatching on the version byte. Knows about
/// v2 (no TQ+) and v3 (with TQ+); anything else errors.
fn read_core_versioned<R: Read>(
    r: &mut R,
    version: u8,
    expected: u8,
    label: &str,
) -> io::Result<CoreLoad> {
    match version {
        2 => read_core_v2(r),
        3 => read_core_v3(r),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported {label} format version: {version} (this build \
                 supports versions 2 and {expected})",
            ),
        )),
    }
}

/// v2: header + codes + scales. Returns empty TQ+ vectors (identity calibration).
fn read_core_v2<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;
    Ok((bit_width, dim, n_vectors, packed_codes, scales, Vec::new(), Vec::new()))
}

/// v3: header + codes + scales + TQ+ trailer.
fn read_core_v3<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;

    let (tqplus_shift, tqplus_scale) = read_tqplus_trailer(r, dim)?;

    Ok((bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale))
}

fn read_tqplus_trailer<R: Read>(r: &mut R, dim: usize) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let mut n_calib_bytes = [0u8; 4];
    r.read_exact(&mut n_calib_bytes)?;
    let n_calib = u32::from_le_bytes(n_calib_bytes) as usize;
    if n_calib != 0 && n_calib != dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid TQ+ n_calib {n_calib}: must be 0 or equal to dim {dim}"),
        ));
    }
    let tqplus_shift = read_f32_array(r, n_calib)?;
    let tqplus_scale = read_f32_array(r, n_calib)?;
    Ok((tqplus_shift, tqplus_scale))
}

fn read_header_codes_scales<R: Read>(
    r: &mut R,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let (bit_width, dim, n_vectors) = read_validated_header(r)?;

    // The validated sizes below are safe to use for allocation.
    let packed_bytes = (dim / 8)
        .checked_mul(bit_width)
        .and_then(|x| x.checked_mul(n_vectors))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "packed code size overflows usize")
        })?;
    let packed_codes = read_exact_vec(r, packed_bytes)?;

    let scales = read_f32_array(r, n_vectors)?;
    Ok((bit_width, dim, n_vectors, packed_codes, scales))
}

fn read_validated_header<R: Read>(r: &mut R) -> io::Result<(usize, usize, usize)> {
    let mut header = [0u8; CORE_HEADER_SIZE];
    r.read_exact(&mut header)?;
    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

    // Validate header fields before allocating anything. The constructors
    // (`new`/`add_2d`) enforce these invariants, but the load path bypasses
    // them — so an untrusted file could otherwise smuggle a `bit_width` that
    // divides-by-zero in `pack::repack` (0 or >8), a `bit_width` of 5..7 that
    // silently passes `from_parts`'s length check and returns wrong scores,
    // or a `dim` that isn't a multiple of 8 (the bit-plane layout is
    // undefined for it and the size formulas diverge → panic). `dim == 0` is
    // the lazy-index sentinel and is only valid alongside `n_vectors == 0`.
    if !matches!(bit_width, 2 | 3 | 4 | 8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid bit_width {bit_width}: must be 2, 3, 4, or 8"),
        ));
    }
    if dim == 0 {
        if n_vectors != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dim 0 (lazy sentinel) requires n_vectors 0, got {n_vectors}"),
            ));
        }
    } else if dim % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid dim {dim}: must be a multiple of 8"),
        ));
    } else if dim > crate::MAX_DIM {
        // Bound the lazily-built dim×dim rotation matrix: a tiny file can
        // declare a huge dim and drive a multi-GB allocation on first search.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid dim {dim}: exceeds maximum {}", crate::MAX_DIM),
        ));
    }

    Ok((bit_width, dim, n_vectors))
}

/// Read exactly `n` bytes without pre-allocating `n` up front. A malicious
/// header can declare a multi-gigabyte length from a tiny file; `read_to_end`
/// on a `take`-limited reader grows the buffer only to the bytes actually
/// present, so we never reserve the attacker's claimed size before confirming
/// the data exists. The length check then rejects a truncated file cleanly.
fn read_exact_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let read = r.take(n as u64).read_to_end(&mut buf)?;
    if read != n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("truncated file: expected {n} bytes, got {read}"),
        ));
    }
    Ok(buf)
}

fn read_f32_array<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let n_bytes = n
        .checked_mul(4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "f32 array size overflows usize"))?;
    let bytes = read_exact_vec(r, n_bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
