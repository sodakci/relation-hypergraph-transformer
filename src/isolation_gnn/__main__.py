"""无需安装 console script 也可使用的统一命令行入口。"""

from __future__ import annotations

import argparse

from .cli import inspect_veristrong, label_veristrong, predict_veristrong


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m isolation_gnn")
    parser.add_argument("command", choices=("inspect", "predict", "label"))
    args, remaining = parser.parse_known_args()
    if args.command == "inspect":
        inspect_veristrong(remaining)
    elif args.command == "predict":
        predict_veristrong(remaining)
    else:
        label_veristrong(remaining)


if __name__ == "__main__":
    main()
