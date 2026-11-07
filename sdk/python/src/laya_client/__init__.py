"""Legacy import alias; use veya_client in new applications."""
from veya_client._compat import install_alias

install_alias(__name__, "veya_client")
