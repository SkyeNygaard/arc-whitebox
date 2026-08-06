# Dual-engine audit status

The theorem-critical numerical certificate has two complete directed-rounding implementations:

- CPython exact rationals plus Decimal/libmpdec;
- direct-C GMP exact rationals plus MPFR.

Run `strengthening/mpfr_full_replay/run_mpfr_replay.sh` to compile with GCC and Clang and regenerate the independent outputs.

See `strengthening/mpfr_full_replay/MPFR_REPLAY_REPORT.md` and `strengthening/docs/DECISION.md`.
