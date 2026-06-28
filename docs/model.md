# Model audit and scientific basis

## Scope

The simulator is a transparent scenario engine for adults performing resistance training. It is not a validated individual prediction model, a body-fat measurement tool, or a substitute for medical or dietetic assessment.

## Why the original behavior was changed

### 1. Forbes was being used outside its defensible role

The original dashboard used `10.4 / (10.4 + fat mass)` as the fraction of surplus weight assigned to lean mass and described it as anabolic efficiency. The Forbes relation describes an empirical relationship between fat mass and fat-free mass changes. Hall later formalized its differential form for body-composition dynamics. It does not establish that a leaner resistance-trained person converts a fixed calorie surplus into muscle at that rate.

The revised model uses the relation only as a **population baseline for weight-loss partitioning**:

\[
p_{FFM,baseline} = \frac{10.4}{10.4 + FM}
\]

It then exposes an explicit FFM-loss-protection assumption:

\[
p_{FFM,adjusted} = p_{FFM,baseline}(1 - protection)
\]

This protection term is a scenario parameter, not a validated universal equation.

### 2. The Alpert value was being treated as a safety boundary

Alpert's estimate concerns a theoretical maximum rate of energy transfer from adipose stores during hypophagia. It was not developed as a clinical “safe deficit” threshold, and crossing it does not prove an abrupt switch to muscle catabolism. The revised app therefore removes the binary safe/unsafe law.

Instead, it displays modeled weekly weight-loss rate against a user-selected planning ceiling. This is intentionally labeled a heuristic rather than a biological limit.

### 3. Energy was converted to tissue incorrectly

The original model divided energy by `7,700 kcal/kg` and then split the resulting mass. Fat tissue and FFM have very different effective energy densities. The revised cut calculation first determines the expected mass partition and then uses a weighted energy density:

\[
\rho_{effective} = p_{FFM}\rho_{FFM} + (1-p_{FFM})\rho_{FM}
\]

\[
\Delta W = \frac{deficit}{\rho_{effective}}
\]

Current constants:

- Fat mass: `9,400 kcal/kg`
- Fat-free mass: `1,800 kcal/kg`

These are approximations used in dynamic body-weight models; they do not capture all energetic costs, adaptive thermogenesis, or short-term fluid shifts.

### 4. FFM was mislabeled as muscle

A two-compartment model can estimate only fat mass and fat-free mass. FFM includes skeletal muscle but also body water, glycogen, organs, bone-associated components, and connective tissue. Every chart and field now uses **FFM** terminology.

### 5. “Bulk staleness” and geometric cycle scaling lacked validation

There is evidence that training status, program quality, genetics, recovery, protein intake, and energy availability affect hypertrophy. There is no validated law showing a fixed linear decline in anabolic efficiency for every week spent in surplus, nor a biological basis for geometrically increasing cycle duration. Both rules were removed.

## Revised bulk model

The model asks for an editable FFM-gain ceiling in `% body weight/month`. Daily potential gain is scaled by training adherence and a bounded protein-support factor:

\[
FFM_{potential/day} = W \times rate_{month} / 30.44 \times adherence \times protein\ support
\]

The energy surplus limits that gain; remaining modeled surplus is assigned to fat mass. This is still a simplification. It is preferable to the prior Forbes-based surplus partition because it exposes the uncertain hypertrophy assumption rather than deriving it from body-fat percentage.

A 2023 trial in resistance-trained lifters found that faster body-mass gain from larger intended surpluses primarily increased skinfold thickness, while evidence for superior hypertrophy was limited. This supports conservative, user-adjustable surplus scenarios rather than an automatic “leaner equals more efficient” law.

## Revised cut model

The model uses the Forbes/Hall relation as a baseline, then reduces modeled FFM loss through the explicit `lean_loss_protection` parameter. This reflects that resistance training, protein intake, training status, deficit size, and recovery can materially alter FFM retention, but available research does not justify a single exact individual formula.

The model also reports `% body weight/week` because rate of loss is more interpretable for planning than a theoretical adipose-energy ceiling. Slower loss has preserved lean mass and performance better than faster loss in some athlete research, while high protein plus intense training has produced markedly better FFM outcomes than lower protein under a large deficit.

## Sensitivity envelope

The app varies:

- starting body-fat estimate,
- FFM-gain response,
- FFM-loss protection.

The minimum and maximum trajectories form a sensitivity envelope. It is **not** a probabilistic confidence or prediction interval.

## Important omissions

The model does not currently simulate:

- adaptive thermogenesis and changing maintenance expenditure,
- non-exercise activity changes,
- water, glycogen, sodium, creatine, or menstrual-cycle effects,
- age-, sex-, drug-, or disease-specific physiology,
- training volume, proximity to failure, exercise selection, or sleep,
- body-fat measurement bias and correlated errors beyond the selected range,
- explicit skeletal-muscle mass.

## References

1. Forbes GB. *Human Body Composition: Growth, Aging, Nutrition, and Activity.* Springer; 1987.
2. Hall KD. Body fat and fat-free mass inter-relationships: Forbes's theory revisited. **Br J Nutr.** 2007;97(6):1059-1063. doi:10.1017/S0007114507691946.
3. Alpert SS. A limit on the energy transfer rate from the human fat store in hypophagia. **J Theor Biol.** 2005;233(1):1-13. doi:10.1016/j.jtbi.2004.08.029.
4. Chow CC, Hall KD. The dynamics of human body weight change. **PLoS Comput Biol.** 2008;4(3):e1000045. doi:10.1371/journal.pcbi.1000045.
5. Hall KD, et al. Quantification of the effect of energy imbalance on bodyweight. **Lancet.** 2011;378(9793):826-837. doi:10.1016/S0140-6736(11)60812-X.
6. Helms ER, et al. Effect of small and large energy surpluses on strength, muscle, and skinfold thickness in resistance-trained individuals. **Sports Med Open.** 2023;9:102. doi:10.1186/s40798-023-00651-y.
7. Garthe I, et al. Effect of two different weight-loss rates on body composition and strength and power-related performance in elite athletes. **Int J Sport Nutr Exerc Metab.** 2011;21(2):97-104. doi:10.1123/ijsnem.21.2.97.
8. Longland TM, et al. Higher compared with lower dietary protein during an energy deficit combined with intense exercise promotes greater lean mass gain and fat mass loss. **Am J Clin Nutr.** 2016;103(3):738-746. doi:10.3945/ajcn.115.119339.
9. Morton RW, et al. A systematic review, meta-analysis and meta-regression of protein supplementation on resistance-training adaptations. **Br J Sports Med.** 2018;52(6):376-384. doi:10.1136/bjsports-2017-097608.
