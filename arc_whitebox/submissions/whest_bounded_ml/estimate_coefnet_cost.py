#!/usr/bin/env python3
"""Simple inference cost estimate for the pairwise coefficient network."""
from __future__ import annotations
import argparse, json

def main():
    p=argparse.ArgumentParser();p.add_argument('--hidden',type=int,default=32);p.add_argument('--width',type=int,default=256);p.add_argument('--layers',type=int,default=31);a=p.parse_args()
    pairs=a.width*(a.width-1)//2
    macs_per_pair=5*a.hidden+a.hidden*a.hidden+2*a.hidden
    macs=pairs*a.layers*macs_per_pair
    params=(5*a.hidden+a.hidden)+(a.hidden*a.hidden+a.hidden)+(2*a.hidden+2)
    print(json.dumps({'hidden':a.hidden,'parameters':params,'pairs_per_layer':pairs,'layers':a.layers,'macs_per_pair':macs_per_pair,'total_macs':macs,'two_flop_fma_equivalent':2*macs},indent=2))
if __name__=='__main__':main()
