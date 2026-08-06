#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('test_result',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=json.loads(a.test_result.read_text())
entries=list(r['hybrid'].values());assert len(entries)==1,'expected exactly one held-out configuration';e=entries[0];s=e['summary']
checks={'gain_at_least_1_25':s['gain_vs_upstream']>=1.25,'win_fraction_at_least_75pct':s['fraction_mlps_improved']>=.75,
        'guard_activation_below_25pct':s.get('fraction_guard_activated',0)<=.25,'worst_mlp_not_catastrophic':s.get('worst_mlp_gain',0)>=.5}
verdict='PORT TO FLOPSCOPE' if all(checks.values()) else ('BORDERLINE' if s['gain_vs_upstream']>=1.05 else 'STOP')
out={'verdict':verdict,'checks':checks,'summary':s,'config':e['config'],'calibration':r.get('calibration')}
text=json.dumps(out,indent=2);print(text)
if a.output:a.output.write_text(text)
