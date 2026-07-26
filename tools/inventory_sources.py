#!/usr/bin/env python3
"""Thin wrapper for ``dynamic-robot-dataset inventory``."""

from dynamic_robot_dataset.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["inventory", *__import__("sys").argv[1:]]))

