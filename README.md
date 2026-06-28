# Body Composition Scenario Lab

A lightweight Streamlit dashboard for exploring **body-composition scenarios** across bulk, cut, and maintenance phases.

The project does not claim to predict an individual's future muscle mass. It models a two-compartment system—fat mass (FM) and fat-free mass (FFM)—and makes uncertain assumptions visible and editable.

## What changed from the original model

- Forbes is retained only as a population-level baseline for partitioning **weight loss**. It is no longer treated as a muscle-gain or “anabolic efficiency” law.
- The Alpert estimate is no longer used as a binary safe-deficit threshold.
- Fat and FFM use separate energy densities instead of converting all energy through `7,700 kcal/kg`.
- “Lean mass” is no longer labeled as muscle.
- Unsupported linear bulk “staleness” and geometric cycle scaling were removed.
- Training, protein, FFM-gain capacity, FFM-loss protection, and measurement uncertainty are explicit assumptions.
- A sensitivity envelope replaces the old single falsely precise trajectory.
- The dependency list was reduced from a frozen environment dump to three runtime packages.
- The model is now pure, typed, testable Python; Streamlit is only the presentation layer.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e .
streamlit run app.py
```

## Test

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

```text
app.py                  Streamlit controls and Plotly charts
bodycomp.py             Pure simulation model and sensitivity analysis
tests/test_bodycomp.py  Behavioral tests
docs/model.md           Equations, evidence, assumptions, and limitations
```

## Interpretation

This is a planning and sensitivity tool, not medical software. FFM includes water, glycogen, organs, connective tissue, and other non-fat components; it is not equivalent to skeletal muscle. The shaded ranges are sensitivity envelopes, not confidence intervals.

See [docs/model.md](docs/model.md) for the model audit and literature basis.
