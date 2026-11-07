"""MCP service using the canonical Veya tools and the ONNX runtime by default."""
from .._runtime.mcp.server import main, server

__all__ = ["main", "server"]


if __name__ == "__main__":
    main()
