# ON/OFF Beam Subtraction — spatial RFI removal + KLT detection

A **simulated** end-to-end pipeline for finding a weak signal of interest (SOI) toward a
target while removing interference (RFI) that arrives from *other* directions. It forms one
**ON** beam on the target plus a ring of **OFF** beams around it, uses the OFF beams as an
interference reference, subtracts that interference from the ON beam, and runs the KLT
detector on the cleaned beam.

It is the *literal* ON/OFF-beam counterpart to the covariance-level nulling primitives that
also live in `helpers.py`: both reach the same goal (a target beam with the interference
removed), one by explicit OFF-beam subtraction, the other by nulling at the antenna
covariance.

---

## The pipeline

Given multi-antenna baseband voltages for one frequency channel:

1. **Generate** the multi-antenna data — a target SOI plus one or more interferers from
   chosen directions (geometry reused from `../MAA_Simulated/`).
2. **Beamform** to an **ON** sky position (the target) with a ring of **OFF** beams around it.
   The ON beam has full gain on the SOI and only sidelobe gain on the RFI; the OFF beams are
   the opposite — that asymmetry is what makes them a usable interference reference.
3. **MR-KLT on the OFF beams** — the interference is the common mode across them, so it lands
   in the dominant eigenvector(s) (`seti_klt.KLT.apply_mrklt`).
4. **Reconstruct** the interference waveform(s) from the OFF-beam eigenvectors.
5. **Subtract** the reconstruction from the ON beam by least squares (the classic
   reference-channel canceller: the OFF beams give the interference *shape*, the fit gives its
   *amplitude and phase* in the ON beam).
6. **C-KLT on the cleaned ON beam** (`seti_klt.KLT.apply_cklt`) to surface the SOI. Before
   cleaning the detector locks onto the RFI; after, it locks onto the SOI.

### Automatic RFI counting

The number of interferers can be found **from the data** instead of being fixed
(`n_off_modes="auto"`), using the **Kritchman & Nadler** rank estimator (`seti_klt.KN_rankEst`)
on the OFF-beam eigenvalues. Two corrections make this work in this setting:

* **Whitening.** Beam noise is *coloured*: its covariance is `σ²·(Wᴴ_off W_off)`, not `σ²·I`,
  because overlapping OFF beams share antennas. K&N assumes a flat noise floor, so we feed it
  the **generalized** eigenvalues of `(R_beam, Wᴴ_off W_off)`, which collapses the noise
  eigenvalues back to a single value and lets K&N count only the true RFI.
* **SOI-leakage exclusion.** The OFF beams' sidelobes pick up a little of the target, so the
  SOI appears as one extra detected mode. We identify it by its overlap with the known target
  signature (`Wᴴ_off · w_ON`) and drop it from the reference, so the subtraction never eats the
  signal we are hunting.

The subtraction is a **multi-reference** least squares (one complex coefficient per detected
RFI), so several interferers are each removed with their own amplitude/phase.

---

## Folder layout

| File | Role |
|---|---|
| `ONOFF_BeamSubtraction.ipynb` | The pipeline, step by step, with diagnostics, a cost benchmark, a parameter playground, and a robustness stress-test. |
| `helpers.py` | Reusable primitives: directional simulation, beamforming, the ON/OFF subtraction pipeline, the one-call experiment harness, and the stress-test runner. |
| `README.md` | This file. |
| `outputs/` | Scratch outputs (git-ignored). |

The *analysis* is not re-implemented here: the notebook calls `seti_klt` (`KLT`,
`apply_mrklt`, `apply_cklt`, `KN_rankEst`) directly, and the array geometry is reused from
`../MAA_Simulated/helpers.py`.

---

## Notebook overview (`ONOFF_BeamSubtraction.ipynb`)

* **Steps 1–6** — the pipeline above, each with a figure (beam grid, 2-D beam pattern,
  MR-KLT eigenspectrum, before/after spectra, the C-KLT lock).
* **Validation** — scores against the injected truth (RFI suppression, SOI loss) and the
  OFF-ring trade-off.
* **§7 Computational cost** — per-stage FLOP breakdown and wall-clock (analysis only, no
  generation), with empirical scaling: the cost is ~linear in the sample count and nearly
  independent of the antenna count.
* **§8 Playground** — a single `PARAMS` block exposes every knob; `run_experiment(...)` runs
  the whole pipeline and `experiment_report(...)` prints/plots the diagnostics. Edit and re-run
  to try any physical situation.
* **§9 Stress-test** — robustness across randomised scenarios: count accuracy and SOI loss vs
  RFI level, number of interferers, OFF-ring radius, and the K&N threshold.

---

## `helpers.py` at a glance

**Simulation (reuses the `MAA_Simulated` geometry)**
* `SimConfig` — array/band parameters.
* `build_geometry`, `steering`, `offset_direction` — antenna layout, steering vectors,
  sky offsets.
* `simulate_cube` — synthesise a `(T, F, A)` antenna cube from directional sources + noise.

**Beamforming**
* `beam_grid` — ON beam + ring of OFF beams (distortionless weights).
* `beamform`, `beam_response`, `beam_pattern`, `beam_pattern_2d` — apply weights, probe gains,
  and map the beam over the sky.

**ON/OFF pipeline**
* `ls_subtract` — reference-channel least-squares subtraction (scalar or multi-reference).
* `process_channel` — the full ON/OFF → MR-KLT → subtract → C-KLT pipeline for one channel,
  including automatic K&N counting (`n_off_modes="auto"`), whitening, and SOI-mode exclusion.

**Harness & tests**
* `run_experiment` / `experiment_report` — one-call experiment from a parameter block, with a
  full diagnostics report.
* `stress_point` — average many randomised realisations of a scenario for the robustness sweeps.

*(`helpers.py` also provides an alternative covariance-level nulling family —
`antenna_cov`, `eig_desc`, `interference_subspace`, `null_weights`, `mvdr_weights` — that
reaches the same goal without forming explicit OFF beams.)*

---

## Running it

Run the cells top to bottom. To explore, edit the `PARAMS` block in **§8** and re-run the two
playground cells; sources are given as **offsets from the target** with a role
(`"signal"` / `"interferer"`), a fine bin, and a level.

```python
res = run_experiment(**PARAMS, logger=quiet)
experiment_report(res)
```

---

## Known limitations / generality

The stress-test (§9) maps where the method holds and breaks in simulation:

* robust across RFI strength and OFF-ring radius; a **wider OFF ring reduces SOI self-subtraction**;
* keep **`n_off` comfortably larger than the RFI count** — accuracy degrades once the number of
  interferers approaches the number of OFF beams;
* **`kn_alpha ≈ 0.5 %`** is the sweet spot (looser values over-detect);
* a residual few-percent miss rate comes from interferers landing between OFF beams.

The main gap to **real data** is the white-noise assumption behind the whitening
(`G = Wᴴ_off W_off` is exact only for uncorrelated thermal antenna noise). On real recordings
the noise is coloured by the sky, calibration, and cross-talk, so a robust deployment would
whiten with a **noise covariance estimated from the data** (RFI-free channels/times) rather than
the analytic form.

## Related material

* `../MAA_Simulated/` — the multi-antenna simulation whose geometry this study reuses.
* `seti_klt` — the KLT library (`KLT`, `apply_mrklt`, `apply_cklt`, `KN_rankEst`).
