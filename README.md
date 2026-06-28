# Body Composition Simulator

A lightweight Streamlit dashboard for exploring plausible long-term changes in body weight, fat mass, fat-free mass and energy expenditure across cutting and gaining phases.

The application uses metric units throughout: kilograms, percentages, grams per kilogram per day, weeks, months and kcal/day.

## Revised model

The previous implementation treated several hypotheses as hard laws. The updated model now:

- uses separate effective energy densities for fat-mass and fat-free-mass change;
- uses Cunningham resting metabolic rate from fat-free mass;
- updates expenditure as body weight and fat-free mass change;
- includes gradual adaptive thermogenesis;
- uses Forbes' relationship as a baseline for the composition of weight change, not as a pure muscle-gain score;
- modifies cut outcomes using protein intake, resistance-training quality and deficit severity;
- treats the Alpert estimate as a continuous risk signal rather than an abrupt cutoff;
- caps projected fat-free-mass gain with explicit training-status priors;
- preserves the exact starting state as day zero;
- labels fat-free mass correctly instead of calling it all muscle.

## Installation

```bash
git clone https://github.com/Znypr/BodyCompSimulator.git
cd BodyCompSimulator
python -m venv .venv
pip install -r requirements.txt
streamlit run main.py
```

## Tests

```bash
pip install pytest
pytest -q
```

The tests cover baseline preservation, energy-balance behaviour, lean-mass retention, training-status gain caps, continuous risk behaviour and metric-unit outputs.

## Limitations

This is a scenario-planning tool, not a clinical prediction engine. Individual responses vary because of measurement error, fluid and glycogen changes, training stimulus, sleep, adherence and genetics.

Fat-free mass is not identical to skeletal muscle. It also includes water, glycogen, organs, bone and other non-fat tissues. Short-term scale changes can therefore differ from the projected tissue trend.

A maintenance intake calibrated from several weeks of measured intake and body-weight data is preferable to an equation-based estimate.

The training-status gain caps are transparent modelling priors designed to prevent impossible outputs. They are not universal biological ceilings.

## Scientific basis

Primary and foundational sources:

1. Hall KD, et al. Quantification of the effect of energy imbalance on bodyweight. Lancet. 2011. DOI: 10.1016/S0140-6736(11)60812-X
2. Chow CC, Hall KD. The dynamics of human body weight change. PLoS Computational Biology. 2008. DOI: 10.1371/journal.pcbi.1000045
3. Hall KD. What is the required energy deficit per unit weight loss? International Journal of Obesity. 2008. DOI: 10.1038/sj.ijo.0803720
4. Cunningham JJ. A reanalysis of the factors influencing basal metabolic rate in normal adults. American Journal of Clinical Nutrition. 1980. DOI: 10.1093/ajcn/33.11.2372
5. Alpert SS. A limit on the energy transfer rate from the human fat store in hypophagia. Journal of Theoretical Biology. 2005. DOI: 10.1016/j.jtbi.2004.08.029
6. Longland TM, et al. Higher compared with lower dietary protein during an energy deficit combined with intense exercise promotes greater lean mass gain and fat mass loss. American Journal of Clinical Nutrition. 2016. DOI: 10.3945/ajcn.115.119339
7. Morton RW, et al. Protein supplementation and resistance-training-induced gains in muscle mass and strength. British Journal of Sports Medicine. 2018. DOI: 10.1136/bjsports-2017-097608

## Project structure

```text
main.py              Streamlit interface and charts
logic.py             Simulation model
style.css            Responsive styling
tests/test_logic.py  Deterministic model tests
```

## License

MIT
