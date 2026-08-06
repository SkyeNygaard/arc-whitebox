# V5.1 release corrections

V5.1 changes release metadata and traceability, not the theorem or proof arithmetic.

- Corrects the audit count from 31 to 32 canonical files.
- Adds all 23 deterministic curvature chunks to the manifest.
- Replaces “immutable manifest” language with “fixed verification manifest”; authenticity is supplied by the archive SHA-256 sidecar.
- Pins `mpmath==1.3.0` and records the tested runtime.
- Adds a multi-OS, multi-Python CI workflow. The workflow configuration is included; this local audit executed only Linux/CPython 3.13.5.
- Retains the one-sided theorem JSON. V4 or other stale artifacts assigning a positive lower bound to Kerdock suboptimality are not part of this release and must be quarantined.
