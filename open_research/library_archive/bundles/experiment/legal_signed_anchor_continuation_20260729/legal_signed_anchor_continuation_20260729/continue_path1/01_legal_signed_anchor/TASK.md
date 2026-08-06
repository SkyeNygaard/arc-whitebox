# Path 1 — Legal signed-anchor estimation

Test a legal estimator of the frozen K32 lower-order anchor through the direct final-output radial-Hermite control, using all 129 Kerdock bases.

## Frozen gates

- Development candidate/base < 0.75.
- Promotion candidate/base <= 0.595, preferably <= 0.537.
- Positive adjusted score.
- Worst network approximately <= 1.10–1.15.
- Complete added compute < 14B.

## Tested primary

Reanchored structured-pilot Gaussianization defect recurrence:

- two disjoint two-basis pilots embedded in the full 129-basis cloud;
- one shared full covariance defect state;
- layerwise source estimated as Gaussian closure around the observed Kerdock preactivation law minus observed pilot post-ReLU covariance;
- selected target means, marginal second moments, and row-direction pair moments only;
- direct final-output control with 128 frozen sample-row radial-Hermite probes;
- no evaluation reference available to candidate construction.

## Bounded rescue

A three-feature ridge predicts a small signed scale using only covariance-source norm, late growth, and pilot disagreement. Hyperparameters are selected by leave-one-network-out replay on tuning networks; validation is opened once.
