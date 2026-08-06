from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
# Oracle ladder
x=pd.read_csv(root/'results/TEST4_ARCHIVED_ORACLE_LADDER.csv')
plt.figure(figsize=(7,4.5)); plt.plot(x.layer,x.aggregate_gain_fraction,marker='o'); plt.xlabel('Oracle replacement layer'); plt.ylabel('Fraction of final MSE removed'); plt.title('Archived layer-mean oracle ladder (screen N=8)'); plt.grid(True,alpha=.25); plt.tight_layout(); plt.savefig(root/'figures/TEST4_ORACLE_LADDER.png',dpi=180); plt.close()
# Signed bound curve
s=pd.read_csv(root/'sources/NEGATIVE_MASS_EXCLUSION_CURVE.csv')
plt.figure(figsize=(7,4.5)); plt.loglog(s.target_improvement_percent,s.beta_integer_kerdock_upper,marker='o'); plt.xlabel('Target Kerdock-relative improvement (%)'); plt.ylabel('Necessary negative mass lower bound'); plt.title('Signed-weight certificate is quantitatively weak'); plt.grid(True,which='both',alpha=.25); plt.tight_layout(); plt.savefig(root/'figures/TEST7_SIGNED_WEIGHT_CURVE.png',dpi=180); plt.close()
# Alignment meta
m=pd.read_csv(root/'results/TEST6_STRATIFIED_META_AUDIT.csv').dropna(subset=['cosine'])
plt.figure(figsize=(8,4.8)); y=np.arange(len(m)); plt.scatter(m.cosine,y,s=55); plt.yticks(y,m.family); plt.axvline(0,linewidth=1); plt.xlabel('Measured correction cosine'); plt.title('Finite-width S2 corrections are not collectively zero-alignment'); plt.tight_layout(); plt.savefig(root/'figures/TEST6_ALIGNMENT_MAP.png',dpi=180); plt.close()
