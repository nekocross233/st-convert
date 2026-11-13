//! Minimal IEEE-754 binary16 conversion, matching what `torch.Tensor.half()`
//! does (round to nearest, ties to even).

/// Round `v` right by `shift` bits with round-half-to-even.
#[inline]
fn round_shift_rne(v: u64, shift: u32) -> u64 {
    if shift == 0 {
        return v;
    }
    if shift >= 64 {
        return 0;
    }
    let half = 1u64 << (shift - 1);
    let rem = v & ((1u64 << shift) - 1);
    let mut q = v >> shift;
    if rem > half || (rem == half && (q & 1) == 1) {
        q += 1;
    }
    q
}

/// f32 -> f16 bit pattern (RNE), plus inf/nan handling.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let abs = x & 0x7fff_ffff;

    // NaN / infinity
    if abs >= 0x7f80_0000 {
        if abs == 0x7f80_0000 {
            return sign | 0x7c00;
        }
        let payload = ((abs >> 13) & 0x3ff) as u16;
        return sign | 0x7c00 | if payload != 0 { payload } else { 1 };
    }

    let exp = (abs >> 23) as i32;
    if exp == 0 {
        // f32 subnormal: |v| < 2^-126, far below the smallest f16 subnormal
        return sign;
    }
    let frac = (abs & 0x7f_ffff) as u64;
    let e = exp - 127; // unbiased exponent
    let sig = frac | 0x80_0000; // 24-bit significand, value = sig * 2^(e-23)

    if e > 15 {
        return sign | 0x7c00; // overflow
    }

    if e >= -14 {
        // normal f16: 11 significant bits
        let mut q = round_shift_rne(sig, 13);
        let mut e2 = e + 15;
        if q >= 0x800 {
            q >>= 1;
            e2 += 1;
        }
        if e2 >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((e2 as u16) << 10) | ((q as u16) & 0x3ff);
    }

    // subnormal f16: value = q * 2^-24 with q in [1, 1024]
    let shift = (-(e + 1)) as u32; // e <= -15  =>  shift >= 14
    let q = round_shift_rne(sig, shift);
    // q == 1024 encodes the smallest normal, which is exactly right.
    sign | (q as u16)
}

/// bf16 bit pattern -> f16 bit pattern (exact widening then RNE narrowing).
#[inline]
pub fn bf16_to_f16_bits(bits: u16) -> u16 {
    f32_to_f16_bits(f32::from_bits((bits as u32) << 16))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(x: f32) -> u16 {
        f32_to_f16_bits(x)
    }

    #[test]
    fn specials() {
        assert_eq!(half(0.0), 0x0000);
        assert_eq!(half(-0.0), 0x8000);
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(-2.0), 0xc000);
        assert_eq!(half(f32::INFINITY), 0x7c00);
        assert_eq!(half(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(half(65504.0), 0x7bff);
        assert_eq!(half(65520.0), 0x7c00); // rounds to inf
        assert_eq!(half(6.1035156e-5), 0x0400); // min normal
        assert_eq!(half(5.9604645e-8), 0x0001); // min subnormal
        assert_eq!(half(2.9802322e-8), 0x0000); // halfway, ties to even -> 0
    }
}
