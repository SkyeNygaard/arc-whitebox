#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, re, subprocess, sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
errors=[]
required=[
 'README.md','RELEASE_STATUS.md','REPRODUCIBILITY.md','AI_ASSISTANCE.md',
 'audit/AUDIT_REPORT.md','audit/CHANGELOG_AUDITED_V2.md','audit/RELEASE_GATES.md',
 'papers/Paper_A_Kerdock_Near_Optimality.md','papers/Paper_A_Kerdock_Near_Optimality.pdf','papers/Paper_A_Kerdock_Near_Optimality.docx',
 'papers/Paper_B_Oracle_Headroom_Open_Ledger.md','papers/Paper_B_Oracle_Headroom_Open_Ledger.pdf','papers/Paper_B_Oracle_Headroom_Open_Ledger.docx',
 'evidence/primary_theory/HISTORICAL_STATUS.md',
 'evidence/primary_theory/v5_2/FORMAL_CANONICAL_THEOREM_RECORD_V5_2.json',
 'evidence/primary_theory/signed_replay/INERTIA_STRENGTHENED_FROZEN_WITNESS_VERIFICATION.json',
 'evidence/primary_theory/KERDOCK_RISK_SANITY_CHECK.json',
 'ledger/csv/README.md','ledger/csv/Experiment_Ledger.csv','ledger/csv/Evidence_Registry.csv'
]
for rel in required:
    if not (ROOT/rel).exists(): errors.append(f'missing required: {rel}')

# Manifest must cover every file except itself, with correct hashes.
manifest=ROOT/'release_manifest.csv'
if not manifest.exists():
    errors.append('missing release_manifest.csv')
else:
    rows=list(csv.DictReader(manifest.open(newline='',encoding='utf-8')))
    listed={r['path'] for r in rows}
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=manifest and '.git' not in p.parts}
    if listed!=actual:
        for x in sorted(actual-listed): errors.append(f'unmanifested: {x}')
        for x in sorted(listed-actual): errors.append(f'manifest missing file: {x}')
    for r in rows:
        p=ROOT/r['path']
        if p.exists():
            if int(r['bytes']) != p.stat().st_size: errors.append(f'byte-size mismatch: {r["path"]}')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
                errors.append(f'hash mismatch: {r["path"]}')

# Files whose prose is presented as current public guidance. Historical evidence
# files and ledger rows are intentionally excluded from this stale-wording scan.
public_rels=[
 'README.md','RELEASE_STATUS.md','RELEASE_STRATEGY.md','REPRODUCIBILITY.md','OPEN_PROBLEMS.md',
 'papers/Paper_A_Kerdock_Near_Optimality.md','papers/Paper_A_Kerdock_Near_Optimality.pdf','papers/Paper_A_Kerdock_Near_Optimality.docx',
 'papers/Paper_B_Oracle_Headroom_Open_Ledger.md','papers/Paper_B_Oracle_Headroom_Open_Ledger.pdf','papers/Paper_B_Oracle_Headroom_Open_Ledger.docx',
 'forum/Forum_Post_A_Kerdock_Near_Optimality.md','forum/Forum_Post_B_Open_Experiment_Ledger.md',
 'review/Two_Page_Reviewer_Overview.md','review/Reviewer_Questions.md','review/Northeastern_Outreach_Revised.md',
 'audit/AUDIT_REPORT.md','audit/RELEASE_GATES.md'
]
public=[ROOT/x for x in public_rels]
combined='\n'.join(p.read_text(errors='replace') for p in public if p.exists())

# The unrecovered stronger constant may occur only as explicitly historical.
for line in combined.splitlines():
    if '0.9370605225569535' in line and not any(w in line.lower() for w in ('reported','unrecovered','excluded','historical','not use')):
        errors.append('unqualified unrecovered T70 constant in public text: '+line[:180])
    if '1.0671669288460727' in line and not any(w in line.lower() for w in ('reported','unrecovered','excluded','historical','not use')):
        errors.append('unqualified unrecovered T70 factor in public text: '+line[:180])

# Exact obsolete formulations, while allowing explicit negations/explanations.
for line in combined.splitlines():
    low=line.lower()
    if 'same-cost gain cap' in low or 'same-cost improvement is at most' in low:
        errors.append('stale same-cost theorem wording: '+line[:180])
    if ('equal-cost theorem' in low or 'same-cost theorem' in low) and not any(x in low for x in ('not', 'do not', 'described as', 'sometimes', 'prohibited')):
        errors.append('unqualified equal-cost theorem wording: '+line[:180])
    if ('6.7167% reduction' in low or '6.72% reduction' in low) and not any(x in low for x in ('not','incorrect','wrong')):
        errors.append('factor-to-reduction metric confusion: '+line[:180])

bib=(ROOT/'papers/references.bib').read_text()
if '10.1109/TIT.2020.3000398' in bib: errors.append('stale Can DOI')
if re.search(r'doi\s*=\s*\{[^}]*\.X\}', bib, flags=re.I): errors.append('malformed DOI .X suffix in bibliography')
placeholder='github.com/'+'USERNAME'
if placeholder in combined or placeholder in (ROOT/'CITATION.cff').read_text(): errors.append('placeholder GitHub URL')

# CFF syntax and release metadata.
try:
    cff=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    assert cff['cff-version']=='1.2.0'
    assert cff['date-released']=='2026-08-02'
    assert 'Nygaard' in json.dumps(cff)
except Exception as exc:
    errors.append(f'invalid CITATION.cff: {exc}')

# Canonical replay constants must agree with public headlines.
try:
    signed=json.loads((ROOT/'evidence/primary_theory/signed_replay/INERTIA_STRENGTHENED_FROZEN_WITNESS_VERIFICATION.json').read_text())
    from decimal import Decimal
    frac=Decimal(signed['audited_floor_fraction_of_kerdock_upper'])
    factor=Decimal(signed['audited_maximum_kerdock_over_rule_factor'])
    if abs(frac-Decimal('0.93706016836650839349')) > Decimal('1e-19'): errors.append('audited signed fraction mismatch in replay JSON')
    if abs(factor-Decimal('1.0671673322143324904')) > Decimal('1e-19'): errors.append('audited signed factor mismatch in replay JSON')
    sanity=json.loads((ROOT/'evidence/primary_theory/KERDOCK_RISK_SANITY_CHECK.json').read_text())
    if sanity.get('status')!='PASS': errors.append('Kerdock risk sanity check is not PASS')
    if 'not a proof' not in sanity.get('evidence_class',''): errors.append('sanity check lacks non-proof label')
except Exception as exc:
    errors.append(f'cannot parse canonical replay JSON: {exc}')


# LibreOffice can append a visible '.X' artifact to exported hyperlinks from
# some DOCX constructions.  Scan the final PDFs, not only the Markdown/BibTeX.
for rel in ('papers/Paper_A_Kerdock_Near_Optimality.pdf',
            'papers/Paper_B_Oracle_Headroom_Open_Ledger.pdf',
            'review/Kerdock_External_Review_Packet.pdf'):
    p=ROOT/rel
    if p.exists():
        cp=subprocess.run(['pdftotext',str(p),'-'],capture_output=True,text=True)
        if cp.returncode: errors.append(f'cannot extract final PDF text: {rel}')
        elif '.X' in cp.stdout: errors.append(f'malformed visible .X hyperlink artifact in {rel}')

# Resolve local markdown links/images.
mds=[p for p in ROOT.rglob('*.md') if '.github' not in p.parts and not (len(p.parts) > len(ROOT.parts)+2 and p.relative_to(ROOT).parts[:2] == ('library_archive','bundles'))]
pat=re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
for md in mds:
    for target in pat.findall(md.read_text(errors='replace')):
        if target.startswith(('http://','https://','#','mailto:')): continue
        target=target.split('#',1)[0]
        if not target: continue
        if not (md.parent/target).resolve().exists(): errors.append(f'broken link {md.relative_to(ROOT)} -> {target}')

# Citation keys in both papers must exist.
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
for paper in ('papers/Paper_A_Kerdock_Near_Optimality.md','papers/Paper_B_Oracle_Headroom_Open_Ledger.md'):
    text=(ROOT/paper).read_text()
    for key in re.findall(r'@([A-Za-z0-9_:-]+)',text):
        if key not in keys: errors.append(f'missing bibliography key in {paper}: {key}')

if errors:
    print('STRICT RELEASE CHECK FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)
print('Strict release check passed.')
