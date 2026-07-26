#!/usr/bin/env python3
"""Thin wrapper for ``dynamic-robot-dataset validate``."""

from dynamic_robot_dataset.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *__import__("sys").argv[1:]]))

