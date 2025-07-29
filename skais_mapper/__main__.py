#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""skais-mapper entry points: configure [or conf|c], generate [or gen|g], help [or h]."""

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import skais_mapper


def parse_args(return_parser: bool = False, **kwargs):
    """Parse arguments.

    Args:
        return_parser: If True, returns parser instead of parsed argument dictionary.
        kwargs: Additional keyword arguments for compatibility.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "configure",
            "conf",
            "c",
            "generate",
            "gen",
            "g",
            "help",
            "h",
        ],
        help="Available subcommands: use `skais-mapper [subcommand] -h` for details",
    )
    if return_parser:
        return parser
    args, _ = parser.parse_known_args()
    configs = vars(args)
    return configs


def main():
    """Main entry point for skais-mapper."""
    parser = parse_args(return_parser=True)
    args, _ = parser.parse_known_args()
    options = vars(args)
    if options["mode"]:
        sys.argv.pop(1)
        match options["mode"]:
            case "configure" | "conf" | "c":
                skais_mapper.configure.create()
            case "generate" | "gen" | "g":
                skais_mapper.generate.run()
            case _:
                parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
