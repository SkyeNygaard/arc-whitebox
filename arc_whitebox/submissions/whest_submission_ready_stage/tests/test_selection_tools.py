from pathlib import Path
import json, subprocess, tempfile


def make(path,cal,offset):
    h={}
    for i,m in enumerate([2.0,1.5,1.2]):
        h[f'c{i}']={'config':{'alpha':.3+i*.1,'beta':0.,'gamma':0.,'corr_cap':.999,'x_clip':20.,'residual_clip':.5},
                  'summary':{'final_mean_mse':m+offset,'fraction_mlps_improved':.9,'fraction_guard_activated':.0}}
    path.write_text(json.dumps({'calibration':cal,'hybrid':h}))


def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);a=td/'a.json';b=td/'b.json';out=td/'sel.json';plan=td/'plan.json'
        make(a,None,0.);make(b,'cal.json',-.1)
        subprocess.run(['python',str(root/'select_hybrid_configs.py'),str(a),str(b),'--top-k','2','--output',str(out)],check=True)
        r=json.loads(out.read_text());assert len(r['selected'])==2 and r['selected'][0]['calibration']=='cal.json'
        subprocess.run(['python',str(root/'choose_final_hybrid.py'),str(b),'--output',str(plan)],check=True)
        p=json.loads(plan.read_text());assert p['calibration']=='cal.json'
    print('selection tests passed')
if __name__=='__main__':main()
