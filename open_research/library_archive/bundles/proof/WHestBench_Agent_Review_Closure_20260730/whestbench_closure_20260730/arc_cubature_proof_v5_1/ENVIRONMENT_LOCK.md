# Environment lock and trust base

Proof-critical directed interval arithmetic uses CPython `Fraction`, `int`, and `decimal`/libmpdec. `mpmath==1.3.0` is used only for explicitly labeled independent audits.

Current clean verification environment:

- Linux x86_64, kernel 6.12.13
- CPython 3.13.5, GCC 14.2.0
- libmpdec 2.5.1
- mpmath 1.3.0

The included CI workflow requests Ubuntu and macOS with Python 3.11–3.13. Those remote jobs are configuration, not results; publication should link the completed CI run.
