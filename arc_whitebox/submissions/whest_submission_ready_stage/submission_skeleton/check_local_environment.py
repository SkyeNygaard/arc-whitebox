#!/usr/bin/env python3
from __future__ import annotations
import json,sys

def version(name):
 try:
  m=__import__(name);return {'version':getattr(m,'__version__','unknown'),'file':getattr(m,'__file__','unknown')}
 except Exception as e:return {'error':repr(e)}
print(json.dumps({'python':sys.version,'flopscope':version('flopscope'),'whestbench':version('whestbench')},indent=2))
