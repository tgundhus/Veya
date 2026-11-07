"""MCP (Model Context Protocol) stdio server for Veya (optional extra).

Install the extra to use it:

    pip install "veya[mcp]"

Then run the server:

    veya-mcp-server            # console script
    python -m veya.mcp.server  # equivalent module form

The submodules ``device``, ``remote`` and ``tools`` are importable without the
``mcp`` dependency; only ``server`` requires the extra.
"""

__all__ = ["device", "remote", "server", "tools"]
