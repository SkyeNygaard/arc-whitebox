from __future__ import annotations
import hashlib
from pathlib import Path
from proof_file_list import PROOF_FILES

BASE=Path(__file__).resolve().parent

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    manifest=BASE/'PROOF_MANIFEST.sha256'
    expected={}
    for line in manifest.read_text().splitlines():
        digest,rel=line.split('  ',1);expected[rel]=digest
    assert set(expected)==set(PROOF_FILES), (set(PROOF_FILES)-set(expected),set(expected)-set(PROOF_FILES))
    for rel in PROOF_FILES:
        p=BASE/rel
        assert p.exists(),p
        actual=sha256(p)
        assert actual==expected[rel],f'{rel}: {actual} != {expected[rel]}'
    print(f'manifest verified: {len(PROOF_FILES)} files')

if __name__=='__main__':main()
