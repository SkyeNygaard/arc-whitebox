# Archive provenance

`PROOF_MANIFEST.sha256` checks the fixed internal file set. A separate SHA-256 sidecar is generated for the final ZIP so that archive identity can be cited independently of its contents. The sidecar created in this audit is timestamped in the shared Library, but it is not a third-party public transparency-log anchor. Publication should additionally record the digest in a durable external release or repository tag.
