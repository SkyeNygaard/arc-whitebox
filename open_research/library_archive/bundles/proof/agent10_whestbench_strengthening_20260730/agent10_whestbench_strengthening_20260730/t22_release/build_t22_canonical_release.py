#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
corrected=json.loads((HERE/'FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32(1).json').read_text())
corrected['schema_version']='whestbench.near_optimality.one_sided.v1'
corrected['canonical_status']='CANONICAL_ONE_SIDED'
corrected['supersedes']={
    'artifact':'FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json',
    'reason':'The superseded artifact incorrectly presented the certificate gap as a positive lower bound on actual Kerdock suboptimality. The theorem is one-sided: actual excess may be zero.'
}
corrected['release_requirements']={
    'external_archive_digest_required':True,
    'manifest_wording':'fixed during verification; not independently authenticated unless the release digest is externally anchored',
    'tracked_canonical_files':32,
    'regenerated_intermediate_curvature_chunks':23,
    'intermediate_chunks_individually_in_primary_manifest':False,
    'tested_environment':'CPython 3.13.5 / libmpdec 2.5.1 / mpmath 1.3.0',
    'cross_platform_ci':'pending'
}
# Preserve hash of semantic content before adding the self hash.
canonical=json.dumps(corrected,sort_keys=True,separators=(',',':'))
corrected['semantic_sha256']=hashlib.sha256(canonical.encode()).hexdigest()
out=HERE/'FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32_CANONICAL.json'
out.write_text(json.dumps(corrected,indent=2)+'\n')
print(out)
