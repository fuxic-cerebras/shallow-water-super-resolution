# Review a change

Review the current diff against `CLAUDE.md` and the relevant acceptance criteria.

Prioritize:

1. numerical changes that alter the PDE solution or boundary behavior;
2. data leakage, time/grid misalignment, or normalization leakage;
3. incorrect vector augmentation or channel order;
4. metric aggregation and unit errors;
5. irreproducible configs, seeds, manifests, and checkpoints;
6. missing tests or documentation drift.

Distinguish correctness issues from optional improvements. Cite exact files and explain
how each issue could affect the scientific conclusion. If no blocking issue is found,
state what was tested and what remains unverified.
