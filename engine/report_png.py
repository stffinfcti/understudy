"""PNG scorecard next to each run record. Pillow is a Hermes dependency; if it
is missing the run simply skips the card."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

W, H, SCALE = 1200, 675, 2
BG, PANEL = "#0E0F12", "#16181D"
TEXT, DIM = "#F2F3F5", "#9AA0A6"
VERDICT_COLORS = {"ready": "#3FB27F", "partial": "#E3A43B",
                  "not_ready": "#E5534B", "insufficient": "#7C8591"}
VERDICT_LABELS = {"ready": "Ready", "partial": "Partial",
                  "not_ready": "Not ready", "insufficient": "Not enough data"}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_FONT_BOLD = [
    "/System/Library/Fonts/SFNS-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    for path in (_FONT_BOLD if bold else _FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size * SCALE)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default(size * SCALE)


def _ellipsize(draw, text: str, font, max_w: int) -> str:
    text = str(text)
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…" if text else text


def _pill(draw, x: int, y: int, label: str, color: str, font):
    pad_x, pad_y = 10 * SCALE, 3 * SCALE
    w = int(draw.textlength(label, font=font)) + pad_x * 2
    h = font.size + pad_y * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=color)
    # Vertically center the text inside the pill.
    draw.text((x + pad_x, y + (h - font.size) / 2 - font.size * 0.1),
              label, font=font, fill="#0E0F12")
    return w


def render_png(record: dict, path: Path) -> Optional[Path]:
    """Write the scorecard PNG; returns None when Pillow is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(path)
    img = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    draw = ImageDraw.Draw(img)

    f_label = _font(13, bold=True)
    f_title = _font(26, bold=True)
    f_sub = _font(14)
    f_model = _font(16, bold=True)
    f_big = _font(32, bold=True)
    f_small = _font(13)
    f_pill = _font(12, bold=True)
    f_footer = _font(12)
    f_cost_big = _font(17, bold=True)

    margin = 48 * SCALE
    y = 40 * SCALE
    draw.text((margin, y), "UNDERSTUDY", font=f_label, fill=DIM)
    y += 22 * SCALE
    draw.text((margin, y), "Could a cheaper model do your agent's job?",
              font=f_title, fill=TEXT)
    y += 40 * SCALE
    corpus = record.get("corpus") or {}
    current = record.get("current_model") or {}
    subtitle = (f"{len(record.get('decisions') or [])} real decisions · "
                f"{corpus.get('sessions', '?')} conversations · "
                f"last {corpus.get('window_days', '?')} days · "
                f"current model {current.get('model') or '?'}")
    draw.text((margin, y), subtitle, font=f_sub, fill=DIM)

    candidates = (record.get("candidates") or [])[:4]
    footer_y = H * SCALE - 56 * SCALE
    # Distribute candidate rows over the remaining height (max 4).
    top = y + 42 * SCALE
    bottom = footer_y - 16 * SCALE
    n = max(1, len(candidates))
    slot = min(118 * SCALE, (bottom - top) // max(n, 1))
    row_h = min(104 * SCALE, slot - 12 * SCALE)
    # Vertically center the row block when fewer than 4 candidates.
    block_h = n * slot
    top += max(0, (bottom - top - block_h) // 2)
    right_x = W * SCALE - margin - 320 * SCALE   # cost block, right-aligned column

    for i, cand in enumerate(candidates):
        y = top + i * slot
        draw.rounded_rectangle([margin, y, W * SCALE - margin, y + row_h],
                               radius=14 * SCALE, fill=PANEL)
        inner = margin + 24 * SCALE
        ty = y + 16 * SCALE
        # Model name — full unless it would pass 40% of card width.
        name_w = int(0.40 * (W * SCALE - 2 * margin))
        model = _ellipsize(draw, cand.get("model") or cand.get("spec") or "?", f_model, name_w)
        draw.text((inner, ty), model, font=f_model, fill=TEXT)
        name_px = draw.textlength(model, font=f_model)
        rec = cand.get("recommendation") or {}
        level = rec.get("level") or "insufficient"
        _pill(draw, inner + name_px + 12 * SCALE, ty - 1 * SCALE,
              VERDICT_LABELS.get(level, level),
              VERDICT_COLORS.get(level, "#7C8591"), f_pill)
        # Big percentage + label + misses + CI bar.
        takeover = cand.get("takeover_rate")
        pct_text = f"{takeover * 100:.0f}%" if isinstance(takeover, (int, float)) else "—"
        draw.text((inner, ty + 28 * SCALE), pct_text, font=f_big, fill=TEXT)
        draw.text((inner, ty + 68 * SCALE), "could take over", font=f_small, fill=DIM)
        counts = cand.get("counts") or {}
        critical = counts.get("CRITICAL_MISS", 0) + counts.get("INVALID", 0)
        minor = counts.get("MINOR_MISS", 0)
        info_x = inner + 170 * SCALE
        draw.text((info_x, ty + 36 * SCALE),
                  f"{critical} critical · {minor} minor misses · scored {cand.get('scored', 0)}",
                  font=f_small, fill=DIM)
        bar_x, bar_y = info_x, ty + 62 * SCALE
        bar_w = min(340 * SCALE, right_x - info_x - 40 * SCALE)
        draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 8 * SCALE],
                               radius=4 * SCALE, fill="#262930")
        if isinstance(takeover, (int, float)):
            fill_w = int(bar_w * takeover)
            if fill_w > 0:
                draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + 8 * SCALE],
                                       radius=4 * SCALE, fill="#5E9BD3")
        ci = cand.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo = bar_x + int(bar_w * float(ci[0]))
            hi = bar_x + int(bar_w * float(ci[1]))
            draw.line([lo, bar_y - 5 * SCALE, hi, bar_y - 5 * SCALE],
                      fill=TEXT, width=2 * SCALE)
            draw.line([lo, bar_y - 9 * SCALE, lo, bar_y],
                      fill=TEXT, width=2 * SCALE)
            draw.line([hi, bar_y - 9 * SCALE, hi, bar_y],
                      fill=TEXT, width=2 * SCALE)
        # Cost block on the right, two lines, never truncated. Insufficient-
        # data or unreliable candidates get a dimmed "not enough data".
        has_cost = False
        low_conf = level == "insufficient" or cand.get("unreliable")
        cost_color = DIM if low_conf else TEXT
        if low_conf:
            draw.text((right_x, ty + 36 * SCALE), "not enough data",
                      font=f_small, fill=DIM)
        elif isinstance(cand.get("cost_ratio"), (int, float)):
            draw.text((right_x, ty + 24 * SCALE),
                      f"{cand['cost_ratio'] * 100:.0f}% of the cost",
                      font=f_cost_big, fill=cost_color)
            has_cost = True
        bits2 = []
        if isinstance(cand.get("projected_monthly_usd"), (int, float)):
            bits2.append(f"~${cand['projected_monthly_usd']:.2f}/mo")
        if isinstance(cand.get("savings_monthly_usd"), (int, float)) and cand["savings_monthly_usd"] > 0:
            bits2.append(f"saves ~${cand['savings_monthly_usd']:.2f}/mo")
        if bits2 and not low_conf:
            draw.text((right_x, ty + 52 * SCALE), " · ".join(bits2),
                      font=f_small, fill=DIM)
            has_cost = True
        if not has_cost and not low_conf:
            draw.text((right_x, ty + 36 * SCALE), "cost unknown", font=f_small, fill=DIM)

    baseline = record.get("baseline") or {}
    counts_b = baseline.get("counts") or {}
    matched = sum(counts_b.get(v, 0) for v in ("EXACT", "AGREE", "ALTERNATIVE", "BETTER"))
    left = f"Your current model matched itself on {matched}/{baseline.get('scored', 0)}"
    if isinstance(record.get("spent_usd"), (int, float)):
        left += f" · spent ${record['spent_usd']:.2f}"
    span = record.get("usage_span_days")
    if span and span < 30:
        left += f" · {span:.1f} days of usage" if span < 2 else f" · {int(span)} days of usage"
    draw.text((margin, footer_y), left, font=f_footer, fill=DIM)
    right = "hermes plugins install understudy"
    rw = draw.textlength(right, font=f_footer)
    draw.text((W * SCALE - margin - rw, footer_y), right, font=f_footer, fill=DIM)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
