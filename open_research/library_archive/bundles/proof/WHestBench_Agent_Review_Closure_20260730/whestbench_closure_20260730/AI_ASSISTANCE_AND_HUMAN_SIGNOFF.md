# AI assistance and required human sign-off

The agent reports, code, and this closure package were substantially generated and audited with language models. Agreement among model agents is not independent human peer review.

Before publication:

1. A named human mathematician should inspect and sign off on:
   - T22 theorem statement and inequality directions;
   - T16 sixth-derivative/Hermite argument and interval certificates;
   - T27 algebra and scope;
   - signed negative-mass lemma.
2. A named reproducibility reviewer should run:
   - the complete T22 v5.1 clean-room regeneration;
   - both T16 Python scripts;
   - the independent C++ audit;
   - all manifest verifiers in the pinned CI matrix.
3. A named evidence reviewer should trace each retained empirical table from raw rows through metric code, selection chronology, grouped uncertainty, and final wording.
4. The paper should disclose:
   - which proofs/code/prose were model-generated;
   - what was independently rerun;
   - what remains computer-assisted rather than proof-assistant formalized;
   - which empirical claims were removed because artifacts were missing.
