# Model v2 notes

The revised simulator supports two planning modes:

- Body-fat range: end a bulk at an upper body-fat boundary, end a cut at a lower boundary, and optionally reserve time for a final cut.
- Fixed duration: use manually selected phase lengths.

Corrections made in v2:

- Calories are recalibrated against current expenditure at each phase transition instead of remaining anchored to day-zero maintenance.
- Forbes' curve is no longer used as a direct muscle-gain or muscle-loss law.
- Cut fat-free-mass loss varies with leanness, loss rate, training consistency, protein intake and fat-energy-transfer stress.
- Gain rates are constrained by training status, available energy and diminishing returns.
- The final cut switches to maintenance after reaching its target rather than repeatedly cutting below it.
- The dashboard shows completed cut checkpoints so a timeline ending during a bulk is not mistaken for the final lean outcome.

The application is a scenario planner rather than a clinical predictor. Fat-free mass also includes water, glycogen, organs and bone; it is not synonymous with skeletal muscle.
