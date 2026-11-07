"""Prepare verified replay metadata with the lightweight ONNX runtime."""
from ._runtime.replay import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
