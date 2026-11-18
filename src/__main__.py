"""Command entry point: ``python -m src <command>``.

The package does not run a script on its own. Name the command.
``cleaning`` cleans the review comments. Later scripts register in
:data:`COMMANDS` and dispatch from :func:`main`.
"""

from __future__ import annotations

import sys

COMMANDS: dict[str, str] = {
    "cleaning": "Clean Digikala review comments and write a CSV.",
    "preprocess": "Rewrite cleaned review bodies and write a CSV.",
    "tokenize": "Tokenize preprocessed review bodies and write parquet.",
    "tookabert-v2": "Download, preprocess, train, and score TookaBERT on this machine.",
    "api": "Serve the review checkpoints over HTTP.",
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``python -m src <command>``.

    No command exits with status 2 and lists the commands. ``--help``
    lists them and returns. A known command receives the remaining
    arguments, including its own ``--help``.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help(file=None if args else sys.stderr)
        if not args:
            raise SystemExit(2)
        return
    command, *rest = args
    if command == "cleaning":
        from src.cleaning import main as cleaning_main

        cleaning_main(rest)
        return
    if command == "preprocess":
        from src.preprocess import main as preprocess_main

        preprocess_main(rest)
        return
    if command == "tokenize":
        from src.tokenize import main as tokenize_main

        tokenize_main(rest)
        return
    if command == "tookabert-v2":
        from src.experiments.tookabert_v2 import main as tookabert_main

        tookabert_main(rest)
        return
    if command == "api":
        import uvicorn

        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000)
        return
    print(f"unknown command: {command}", file=sys.stderr)
    _print_help(file=sys.stderr)
    raise SystemExit(2)


def _print_help(*, file) -> None:
    print("usage: python -m src <command> [options]", file=file)
    print(file=file)
    print("commands:", file=file)
    for name, help_text in COMMANDS.items():
        print(f"  {name:<16} {help_text}", file=file)
    print(file=file)
    print("Run python -m src <command> --help for that command's options.", file=file)


if __name__ == "__main__":
    main()
