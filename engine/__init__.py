"""Understudy engine — Hermes-free. Every Hermes touchpoint is injected."""


def write_report_artifacts(record: dict, store) -> None:
    """PNG card + HTML report next to the record; sets card_png/report_html."""
    from . import report_html, report_png
    run_id = record.get("id") or ""
    try:
        png = report_png.render_png(record, store._path(run_id).with_suffix(".png"))
        if png:
            record["card_png"] = str(png)
    except Exception:
        pass
    try:
        html_path = report_html.render_html(record, store._path(run_id).with_suffix(".html"))
        record["report_html"] = str(html_path)
    except Exception:
        pass
