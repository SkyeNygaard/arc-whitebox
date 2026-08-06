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
    lines=[]
    for rel in PROOF_FILES:
        p=BASE/rel
        if not p.exists(): raise FileNotFoundError(p)
        lines.append(f'{sha256(p)}  {rel}')
    (BASE/'PROOF_MANIFEST.sha256').write_text('\n'.join(lines)+'\n')
    print(f'wrote {len(lines)} entries')

if __name__=='__main__':main()
