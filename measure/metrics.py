"""Per-item measurements for Latent Raw shots.

Turns an 8-bit source frame, its 32-bit ACES EXR, the SAM 3.1 item masks and the
synthesized-pixel mask into plain rows (dicts) for the evidence tables.

Every row carries an evidence_id so the agent can cite it ("no row, no claim").
Expanded range is reported as *synthesized*, never "recovered".
"""
import json
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image

# Luminance weights. Source frames are Rec.709 / sRGB; Latent Raw EXRs are ACES.
REC709_Y = np.array([0.2126, 0.7152, 0.0722])
ACES_AP0_Y = np.array([0.3439664498, 0.7281660966, -0.0721325464])  # ACES2065-1
ACES_AP1_Y = np.array([0.2722287168, 0.6740817658, 0.0536895174])   # ACEScg

# A channel at 250 or above counts as clipped. AI video rolls highlights off below 254,
# so a stricter level finds nothing clipped in real clips (Kyo, 26 Oct).
CLIP_LEVEL_8BIT = 250
EPS = 1e-6


def load_8bit(path):
    """8-bit frame as float RGB in 0..1 (still sRGB-encoded)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path, shape):
    """Any greyscale/RGB mask image -> boolean array; white = inside."""
    m = np.asarray(Image.open(path).convert("L"))
    if m.shape != shape:
        raise ValueError(f"mask {path} is {m.shape}, frame is {shape}")
    return m > 127


def load_item_masks(folder, shape):
    """A folder of SAM 3.1 masks, one PNG per item, named by label: car.png, sky.png."""
    return {p.stem: load_mask(p, shape) for p in sorted(Path(folder).glob("*.png"))}


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load_exr(path):
    """Returns (header dict as strings, float32 RGB array) from a scanline EXR."""
    with OpenEXR.File(str(path)) as f:
        header = {k: str(v) for k, v in f.header().items()}
        ch = f.channels()
        if "RGB" in ch:
            rgb = ch["RGB"].pixels
        else:
            rgb = np.stack([ch[c].pixels for c in ("R", "G", "B")], axis=-1)
    return header, rgb.astype(np.float32)


def _aces_y(aces):
    return ACES_AP0_Y if aces == "AP0" else ACES_AP1_Y


def _balance(lin_rgb, m):
    """Colour balance of an item in linear light: mean R/G and B/G. 1.0, 1.0 is neutral."""
    r, g, b = (float(lin_rgb[..., i][m].mean()) for i in range(3))
    return round(r / max(g, EPS), 4), round(b / max(g, EPS), 4)


def exr_header_row(shot_id, frame, path):
    header, rgb = load_exr(path)
    return {
        "evidence_id": f"hdr:{shot_id}:f{frame}",
        "shot_id": shot_id,
        "frame": frame,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "compression": header.get("compression"),
        "chromaticities": header.get("chromaticities"),
        "aces_image_container_flag": header.get("acesImageContainerFlag"),
        "header_json": json.dumps(header),
    }


def frame_range_row(shot_id, frame, src_8bit, exr_rgb, synth_mask, aces="AP0"):
    """Whole-frame dynamic range: what the 8-bit source had vs the EXR."""
    y_hdr = exr_rgb @ _aces_y(aces)
    y_src = srgb_to_linear(src_8bit) @ REC709_Y
    pos = y_hdr[y_hdr > 0]
    lo, hi = (np.percentile(pos, [0.1, 99.9]) if pos.size else (0.0, 0.0))
    return {
        "evidence_id": f"range:{shot_id}:f{frame}",
        "shot_id": shot_id,
        "frame": frame,
        "src_clipped_pct": round(100 * float((np.rint(src_8bit * 255) >= CLIP_LEVEL_8BIT).any(-1).mean()), 3),
        "src_peak_lum": round(float(y_src.max()), 5),
        "hdr_peak_lum": round(float(y_hdr.max()), 5),
        "hdr_mean_lum": round(float(y_hdr.mean()), 5),
        "hdr_range_stops": round(float(np.log2(hi / lo)), 3) if lo > 0 else None,
        "synthesized_pct": round(100 * float(synth_mask.mean()), 3),
    }


def item_rows(shot_id, frame, src_8bit, items, synth_mask, exr_rgb=None, aces="AP0"):
    """One row per SAM item. items = {"sky": bool_mask, "car": bool_mask, ...}."""
    lin = srgb_to_linear(src_8bit)
    y_src = lin @ REC709_Y
    mx, mn = src_8bit.max(-1), src_8bit.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, EPS), 0.0)
    clipped = (np.rint(src_8bit * 255) >= CLIP_LEVEL_8BIT).any(-1)
    y_hdr = exr_rgb @ _aces_y(aces) if exr_rgb is not None else None

    rows = []
    for name, m in items.items():
        if not m.any():
            continue
        rg, bg = _balance(lin, m)
        row = {
            "evidence_id": f"item:{shot_id}:f{frame}:{name}",
            "shot_id": shot_id,
            "frame": frame,
            "item": name,
            "area_pct": round(100 * float(m.mean()), 3),
            "clipped_pct": round(100 * float(clipped[m].mean()), 3),
            "mean_lum": round(float(y_src[m].mean()), 5),
            "peak_lum": round(float(y_src[m].max()), 5),
            "mean_saturation": round(float(sat[m].mean()), 4),
            "r_over_g": rg,
            "b_over_g": bg,
            "synthesized_pct": round(100 * float(synth_mask[m].mean()), 3),
            "hdr_peak_lum": None,
            "hdr_mean_lum": None,
            "hdr_r_over_g": None,
            "hdr_b_over_g": None,
        }
        if y_hdr is not None:
            row["hdr_peak_lum"] = round(float(y_hdr[m].max()), 5)
            row["hdr_mean_lum"] = round(float(y_hdr[m].mean()), 5)
            row["hdr_r_over_g"], row["hdr_b_over_g"] = _balance(exr_rgb, m)
        rows.append(row)
    return rows


def balance_rows(rows_a, rows_b, stage, use_hdr=True):
    """Compare the same items across two clips (or two frames) for the Verifier.

    rows_a / rows_b are item_rows() output. stage is 'before' or 'after' the balance
    pass. Each matched item gets the gap in stops of mean luminance and the gap in
    R/G and B/G. Small gaps mean the clips will cut together.
    """
    lum = "hdr_mean_lum" if use_hdr else "mean_lum"
    rg = "hdr_r_over_g" if use_hdr else "r_over_g"
    bg = "hdr_b_over_g" if use_hdr else "b_over_g"
    by_item = {r["item"]: r for r in rows_b}
    out = []
    for a in rows_a:
        b = by_item.get(a["item"])
        if b is None or a[lum] is None or b[lum] is None:
            continue
        out.append({
            "evidence_id": f"balance:{a['shot_id']}:{b['shot_id']}:{a['item']}:{stage}",
            "shot_a": a["shot_id"],
            "shot_b": b["shot_id"],
            "frame_a": a["frame"],
            "frame_b": b["frame"],
            "item": a["item"],
            "stage": stage,
            "lum_gap_stops": round(float(np.log2(max(b[lum], EPS) / max(a[lum], EPS))), 3),
            "r_over_g_gap": round(b[rg] - a[rg], 4),
            "b_over_g_gap": round(b[bg] - a[bg], 4),
            "cites": [a["evidence_id"], b["evidence_id"]],
        })
    return out
