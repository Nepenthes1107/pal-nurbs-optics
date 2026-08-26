#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    from .version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"
from .basics import *
from .optics import *
# from .solvers import *
