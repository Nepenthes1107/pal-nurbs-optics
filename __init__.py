#!/usr/bin/env python
# coding: utf-8

# In[ ]:


if __package__:
    try:
        from .version import __version__
    except ModuleNotFoundError as exc:
        # The legacy root package has no version.py in this checkout.  Only
        # treat that specific optional module as absent; unrelated import
        # failures must remain visible.
        if exc.name != f"{__package__}.version":
            raise
        __version__ = "0+unknown"
else:
    # Pytest imports the repository root __init__.py as a top-level module on
    # Linux.  Relative imports are invalid in that mode.
    __version__ = "0+unknown"
if __package__:
    from .basics import *
    from .optics import *
    # from .solvers import *
