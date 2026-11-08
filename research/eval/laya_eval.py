"""Compatibility imports for recorded benchmark runners; use ``veya_eval`` for new work."""
if __package__:
    from .veya_eval import *  # noqa: F401,F403
else:
    from veya_eval import *  # noqa: F401,F403
