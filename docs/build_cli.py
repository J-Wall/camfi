#!/usr/bin/env python

import os
import re
import sys
from typing import List

import camfi
import fire


def wrap_line(line: str, n: int) -> List[str]:
    leading_spaces = 0
    for character in line:
        if character == " ":
            leading_spaces += 1
        else:
            break

    words = line.split()
    chars_count = [len(word) + 1 for word in words]

    line_list: List[List[str]] = []
    current_line: List[str] = []

    line_length = leading_spaces - 1
    for word in words:
        line_length += len(word) + 1
        if line_length > n:
            line_list.append(current_line)
            current_line = [word]
            line_length = leading_spaces + len(word)
        else:
            current_line.append(word)

    line_list.append(current_line)

    return [(" " * leading_spaces) + " ".join(line) for line in line_list]


def wrap_file(filename, n):
    with open(filename, "r") as f:
        in_lines = f.readlines()

    out_lines: List[str] = []
    for in_line in in_lines:
        out_lines.extend(wrap_line(in_line, n))

    with open(filename, "w") as f:
        for line in out_lines:
            print(line, file=f)


def fix_commandname(filename, commandname, to_replace=os.path.basename(sys.argv[0])):
    with open(filename, "r") as f:
        help_str = f.read()

    with open(filename, "w") as f:
        f.write(re.sub(to_replace, commandname, help_str))


def main():
    wrap = 74
    os.makedirs("usage/helppages", exist_ok=True)

    commands = [a for a in dir(camfi.AnnotationUtils) if not a.startswith("_")]
    filenames = [f"usage/helppages/{command}.txt" for command in commands]

    # Append " -- --help" to commands
    commands = [command + " -- --help" for command in commands]

    # Also do the main camfi command
    commands.append("-- --help")
    filenames.append("usage/helppages/camfi.txt")

    for command, filename in zip(commands, filenames):
        os.environ["PAGER"] = f'sed "s,\\x1B\\[[0-9;]*[a-zA-Z],,g" > {filename}'
        try:
            fire.Fire(camfi.AnnotationUtils, command)
        except fire.core.FireExit:
            pass
        fix_commandname(filename, "camfi")
        wrap_file(filename, wrap)

    os.environ[
        "PAGER"
    ] = f'sed "s,\\x1B\\[[0-9;]*[a-zA-Z],,g" > usage/helppages/traincamfiannotator.txt'
    try:
        fire.Fire(camfi.train_model, "-- --help")
    except fire.core.FireExit:
        pass
    fix_commandname("usage/helppages/traincamfiannotator.txt", "traincamfiannotator")
    wrap_file("usage/helppages/traincamfiannotator.txt", wrap)

    os.environ[
        "PAGER"
    ] = f'sed "s,\\x1B\\[[0-9;]*[a-zA-Z],,g" > usage/helppages/camfiannotate.txt'
    try:
        fire.Fire(camfi.annotate, "-- --help")
    except fire.core.FireExit:
        pass
    fix_commandname("usage/helppages/camfiannotate.txt", "camfiannotate")
    wrap_file("usage/helppages/camfiannotate.txt", wrap)


if __name__ == "__main__":
    main()
