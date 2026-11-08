"""Live environment lookup with Veya names taking precedence over legacy names."""
import os


def getenv(name, default=None):
    if name.startswith("VEYA_") and name not in os.environ:
        return os.environ.get("LAYA_" + name[5:], default)
    return os.environ.get(name, default)
