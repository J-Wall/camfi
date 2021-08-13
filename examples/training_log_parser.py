from collections.abc import Iterable
from enum import Enum
from math import nan
import re


_epoch_pattern = re.compile(r"\[[0-9]+\]")
_iteration_pattern = re.compile(r"\[[ 0-9]+\/[ 0-9]+\]")
_float_pattern = re.compile(r" [A-Za-z_ ]+: +[0-9\.]+(?: |$)")
_time_pattern = re.compile(
    r"(?: [A-Za-z_ ]+: +)(?:[0-9]{1,2}:){1,2}(?: |[0-9]{1,2}(?:\.[0-9]+)? )"
)


class LineType(Enum):
    ITERATION = 1
    EPOCH = 2
    EOF = 3
    NON_LOGGING_LINE = 4


def parse_epoch(line: str) -> int:
    epoch_match = _epoch_pattern.search(line)
    if epoch_match:
        return int(epoch_match[0].strip("[]"))
    else:
        raise ValueError


def parse_iteration(line: str) -> tuple[int, int]:
    iteration_match = _iteration_pattern.search(line)
    if iteration_match:
        iter_n, iter_tot = tuple(
            int(v) for v in iteration_match[0].strip(" []").split("/")
        )
        return iter_n, iter_tot
    else:
        raise ValueError


def parse_val(val_str: str) -> tuple[str, float]:
    name, val = val_str.strip().split(": ")
    return name, float(val)


def parse_time_s(time_str: str) -> tuple[str, float]:
    name, time = time_str.strip().split(": ")
    hms = time.split(":")
    assert 2 <= len(hms) <= 3
    s = float(hms[-1])
    m = float(hms[-2])
    h = float(hms[-3]) if len(hms) == 3 else 0.0
    tot_s = s + 60 * (m + 60 * h)
    return name, tot_s


def parse_line(line: str) -> tuple[int, int, int, dict[str, float]]:
    epoch = parse_epoch(line)
    iteration, epoch_iterations = parse_iteration(line)
    val_strs = _float_pattern.findall(line)
    time_strs = _time_pattern.findall(line)
    d: dict[str, float] = {}
    for val_str in val_strs:
        name, val = parse_val(val_str)
        d[name] = val
    for time_str in time_strs:
        name, val = parse_time_s(time_str)
        d[name] = val

    return epoch, iteration, epoch_iterations, d


def parse_tot_line(tot_line: str) -> tuple[int, float]:
    epoch = parse_epoch(tot_line)
    time_match = _time_pattern.search(tot_line)
    if time_match:
        _, tot_time = parse_time_s(time_match[0])
    else:
        raise ValueError

    return epoch, tot_time


def line_type(line: str) -> LineType:
    if "Total time" in line:
        return LineType.EPOCH
    elif line.startswith("Epoch"):
        return LineType.ITERATION
    elif line.startswith("Training complete"):
        return LineType.EOF
    return LineType.NON_LOGGING_LINE


def parse_lines(lines: Iterable[str]) -> list[dict[str, list[float]]]:
    epochs: list[dict[str, list[float]]] = []
    vals: dict[str, list[float]] = {"iterations": []}
    n_vals: int = 0
    add_iterations: int = 0
    for line in lines:
        lt = line_type(line)
        if lt == LineType.ITERATION:
            epoch, iteration, epoch_iterations, d = parse_line(line)
            for key in vals.keys() | d.keys():
                if key == "iterations":
                    vals[key].append(float(iteration + add_iterations))
                else:
                    vals.setdefault(key, [nan] * n_vals).append(d.get(key, nan))
            n_vals += 1
        elif lt == LineType.EPOCH:
            add_iterations += epoch_iterations
            epoch, tot_time = parse_tot_line(line)
            epochs.append(vals)
            vals = {"iterations": []}
            n_vals = 0
        elif lt == LineType.EOF:
            break
        elif lt == LineType.NON_LOGGING_LINE:
            pass

    return epochs
