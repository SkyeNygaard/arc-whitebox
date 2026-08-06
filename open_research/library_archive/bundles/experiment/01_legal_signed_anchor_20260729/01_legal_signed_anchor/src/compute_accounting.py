D=256; LAYERS=32; GROUPS=2; ROWS_PER_GROUP=1024; N_FULL=129*512
pilot_cov=GROUPS*LAYERS*(2*ROWS_PER_GROUP*D*D)
transport=LAYERS*(4*D**3)  # two dense matmuls, 2*D^3 FLOPs each
reductions_low=LAYERS*2*(2*N_FULL*D)
reductions_high=LAYERS*2*(3*N_FULL*D)
print({'pilot_cov':pilot_cov,'transport':transport,'reductions_low':reductions_low,'reductions_high':reductions_high,'total_low':pilot_cov+transport+reductions_low,'total_high':pilot_cov+transport+reductions_high})
