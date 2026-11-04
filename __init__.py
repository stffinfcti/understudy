"""Understudy plugin — audition cheaper models on your real Hermes history."""

from __future__ import annotations

from pathlib import Path


def register(ctx):
    from .surfaces import cli as _cli
    from .surfaces import commands as _commands
    from .surfaces import tools as _tools

    ctx.register_tool(
        name="understudy_audition", toolset="understudy",
        schema=_tools.AUDITION_SCHEMA,
        handler=lambda args, **kw: _tools.audition_handler(ctx, args, **kw),
        description="Audition models on your real Hermes history",
    )
    ctx.register_tool(
        name="understudy_report", toolset="understudy",
        schema=_tools.REPORT_SCHEMA,
        handler=lambda args, **kw: _tools.report_handler(ctx, args, **kw),
        description="Read an earlier Understudy audition",
    )
    ctx.register_command(
        "understudy",
        handler=lambda raw, **kw: _commands.handle(ctx, raw),
        description="Audition cheaper models on your real history",
        args_hint="<model…> [--sample N] [--days N] [--budget X] [--yes]",
    )
    ctx.register_cli_command(
        name="understudy",
        help="Audition cheaper models on your real history",
        description="Replay past decisions on candidate models and score them",
        setup_fn=lambda sub: _cli.setup(sub, ctx),
        handler_fn=_cli.make_handler(ctx),
    )
    ctx.register_skill(
        "understudy",
        Path(__file__).resolve().parent / "skills" / "understudy" / "SKILL.md",
        description="Audition cheaper models on real Hermes history before switching",
    )
