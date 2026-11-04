#!/usr/bin/env python3
"""Dev-only: render docs/banner.png (1200x600 @2x) for the plugin catalog.

Reads the real demo run record if present so the right-hand candidate rows
match the shipped scorecard; falls back to the demo numbers baked in below.

    python docs/make_banner.py [path/to/run.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from engine.report_png import (  # noqa: E402
    BG, PANEL, TEXT, DIM, SCALE, VERDICT_COLORS, VERDICT_LABELS,
    _ellipsize, _font, _pill,
)

W, H = 1200, 600

FALLBACK = [
    {"model": "GLM-5.3-Flash", "level": "ready", "takeover": 0.90, "cost": "8% of the cost"},
    {"model": "DeepSeek-V4.1-Flash", "level": "not_ready", "takeover": 0.80, "cost": "6% of the cost"},
    {"model": "Qwen3.5-9B", "level": "not_ready", "takeover": 0.60, "cost": "6% of the cost"},
]


def rows_from_record(path: Path) -> list:
    try:
        record = json.loads(path.read_text())
    except Exception:
        return FALLBACK
    rows = []
    for cand in (record.get("candidates") or [])[:3]:
        level = (cand.get("recommendation") or {}).get("level") or "insufficient"
        takeover = cand.get("takeover_rate")
        cost = (f"{cand['cost_ratio'] * 100:.0f}% of the cost"
                if isinstance(cand.get("cost_ratio"), (int, float)) else "cost unknown")
        rows.append({"model": cand.get("model") or cand.get("spec") or "?",
                     "level": level, "takeover": takeover, "cost": cost})
    return rows or FALLBACK


def main() -> int:
    from PIL import Image, ImageDraw
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "banner-record.json"
    rows = rows_from_record(src)

    img = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    draw = ImageDraw.Draw(img)

    f_kicker = _font(14, bold=True)
    f_title = _font(52, bold=True)
    f_sub = _font(19)
    f_tag = _font(16)
    f_model = _font(15, bold=True)
    f_small = _font(13)
    f_big = _font(24, bold=True)
    f_pill = _font(11, bold=True)

    margin = 56 * SCALE
    mid_x = W * SCALE * 0.46  # right panel starts

    y = 150 * SCALE
    draw.text((margin, y), "HERMES PLUGIN", font=f_kicker, fill=DIM)
    y += 34 * SCALE
    draw.text((margin, y), "Understudy", font=f_title, fill=TEXT)
    y += 74 * SCALE
    draw.text((margin, y), "Audition cheaper models on your real Hermes history.",
              font=f_sub, fill=TEXT)
    y += 30 * SCALE
    draw.text((margin, y), "Know before you switch.", font=f_tag, fill=DIM)

    # Right: three compact candidate rows on one panel.
    px, pw = mid_x, W * SCALE - margin - mid_x
    row_h, gap = 92 * SCALE, 14 * SCALE
    block_h = len(rows) * row_h + (len(rows) - 1) * gap
    y = (H * SCALE - block_h) // 2
    name_w = int(pw * 0.52)
    for row in rows:
        draw.rounded_rectangle([px, y, px + pw, y + row_h], radius=12 * SCALE, fill=PANEL)
        ix = px + 20 * SCALE
        ty = y + 16 * SCALE
        model = _ellipsize(draw, row["model"], f_model, name_w - 60 * SCALE)
        draw.text((ix, ty), model, font=f_model, fill=TEXT)
        w = _pill(draw, ix + draw.textlength(model, font=f_model) + 10 * SCALE,
                  ty - 1 * SCALE, VERDICT_LABELS.get(row["level"], row["level"]),
                  VERDICT_COLORS.get(row["level"], "#7C8591"), f_pill)
        ty2 = ty + 30 * SCALE
        takeover = f"{row['takeover'] * 100:.0f}%" if isinstance(row["takeover"], (int, float)) else "—"
        draw.text((ix, ty2), takeover, font=f_big, fill=TEXT)
        pct_w = draw.textlength(takeover, font=f_big)
        draw.text((ix + pct_w + 10 * SCALE, ty2 + 12 * SCALE), "could take over",
                  font=f_small, fill=DIM)
        draw.text((ix + pct_w + 10 * SCALE, ty2 + 30 * SCALE), row["cost"],
                  font=f_small, fill=DIM)
        y += row_h + gap

    out = HERE / "banner.png"
    img.save(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
