#!/usr/bin/env python3
"""Conservative analytic estimate for the CoefNet-only runtime cost."""
from __future__ import annotations
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=256);p.add_argument('--depth',type=int,default=32);p.add_argument('--hidden',type=int,default=64);p.add_argument('--dtype-rate',type=float,default=1.0);a=p.parse_args()
pairs=a.width*(a.width-1)//2;relu_layers=a.depth-1
# Dense contractions: 2*m*n*k convention. Elementwise feature construction,
# biases and SiLU are shown separately because exact catalog weights vary by
# installed flopscope revision.
matmul_per_pair=2*(5*a.hidden+a.hidden*a.hidden+2*a.hidden)
matmul=relu_layers*pairs*matmul_per_pair*a.dtype_rate
# Very conservative allowance: normalization/features + two sigmoid/SiLU blocks.
elementwise_per_pair=5*10 + a.hidden*20*2 + 2*4
elementwise=relu_layers*pairs*elementwise_per_pair*a.dtype_rate
out={'width':a.width,'depth':a.depth,'hidden':a.hidden,'pairs_per_layer':pairs,'relu_layers':relu_layers,
     'coefnet_matmul_flops':int(matmul),'conservative_elementwise_allowance':int(elementwise),
     'coefnet_total_proxy':int(matmul+elementwise),'phase1_budget':272_000_000_000,
     'budget_fraction_proxy':(matmul+elementwise)/272_000_000_000}
print(json.dumps(out,indent=2))
