import json
import sys
from pathlib import Path


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    # MCP server mode: python -m mathlib_py mcp
    if argv and argv[0] in ("mcp", "serve", "server"):
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0
    # ASCII video mode: python -m mathlib_py ascii --op mul --base 8 --xs 1,2,4,8,100,1000
    if argv and argv[0] in ("ascii", "video", "ascii-video"):
        from .ascii_vid import generate_ascii_video_from_series, play_ascii_video
        from . import MathOp
        op = MathOp.MUL
        base_v = 8
        xs = list(range(1, 13))
        tick = 1.0
        i = 1
        while i < len(argv):
            if argv[i] == "--op" and i + 1 < len(argv):
                op = MathOp.DIV if argv[i + 1].lower() in ("div", "/", "division") else MathOp.MUL
                i += 2
            elif argv[i] == "--base" and i + 1 < len(argv):
                base_v = int(argv[i + 1]); i += 2
            elif argv[i] == "--xs" and i + 1 < len(argv):
                xs = [int(x) for x in argv[i + 1].split(",") if x.strip()]; i += 2
            elif argv[i] == "--tick" and i + 1 < len(argv):
                tick = float(argv[i + 1]); i += 2
            else:
                i += 1
        res = generate_ascii_video_from_series(op, base_v, xs, width=72, height=18, tick=tick)
        print(play_ascii_video(res))
        print("\n" + json.dumps({"css_rule_c": res["css_rule_c"], "memory": res["memory"], "frames": res["frame_count"]}, indent=2))
        return 0

    # Default: GGUF timeline check
    from .gguf_timeline import run_timeline_check
    gguf = Path(argv[0]) if argv else Path("benchmark.gguf")
    if len(argv) > 1:
        years = [int(x) for x in argv[1].split(",") if x.strip()]
    else:
        years = [1, 8, 12, 13, 100, 1000, 8000]
    base_val = 8
    out_path = gguf.with_name(gguf.stem + "_updated" + gguf.suffix)
    result = run_timeline_check(gguf, years, base_value=base_val, update_out_path=out_path)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

