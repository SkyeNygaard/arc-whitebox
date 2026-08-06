"""Force proof scripts to use CPython's independent pure-Python decimal engine."""
from _pydecimal import *  # noqa: F401,F403
from _pydecimal import __version__
try:
    from _pydecimal import __libmpdec_version__
except ImportError:
    __libmpdec_version__ = None
