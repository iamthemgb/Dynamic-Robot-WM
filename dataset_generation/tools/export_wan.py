#!/usr/bin/env python3
"""Thin wrapper for ``dynamic-robot-dataset export-wan``."""

from dynamic_robot_dataset.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["export-wan", *__import__("sys").argv[1:]]))

