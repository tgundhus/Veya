"""HTTP service using the canonical Veya server and the ONNX runtime by default."""
from ._runtime.serve import build_router, create_app, main

__all__ = ["build_router", "create_app", "main"]


if __name__ == "__main__":
    main()
