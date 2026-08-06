"""Integration template, not yet a complete submission.

The remaining required component is a flopscope.numpy port of ARC's factorized
K3 state update. Keep this file beside coefnet_flops.py and the packed model.
"""
from pathlib import Path
from whestbench import BaseEstimator
from coefnet_flops import CoefNet


class Estimator(BaseEstimator):
    def setup(self, ctx):
        here=Path(__file__).parent
        self.coefnet=CoefNet.from_file(here/'coefnet_model.npz')
        # Precompute static pair indices and load validation-selected constants here.

    def predict(self, mlp, budget):
        raise NotImplementedError(
            'Insert the factorized-K3 flopscope port, then call the chunked CoefNet correction after each ReLU.'
        )
