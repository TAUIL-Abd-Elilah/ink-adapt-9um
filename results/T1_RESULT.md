# T1 result (PHerc. 0343): label-free adaptation is no better than the teacher ensemble

Scored 8 Nov 2026 with `code/score_t1.py`. The setup: Nieuwlaar's 18 letter boxes on render66, d = 0, reading side, 1000 letter-bootstrap resamples, the spec in `configs/spec_t1.json`. Everything was fixed in `PREREG.md` (with Amendments 1-7) before any student map existed. The raw scores are in `score_t1_d0.json`, `score_t1_fwd.json` and `score_t1_d1.json` (this folder).

## Verdict

**"No better than the ensemble."** D1 = AUC(student, mean of 3 seeds) - AUC(teach4) = **-0.004**, 95 % CI [-0.032, +0.033]. Neither pre-registered threshold (±0.02 with the CI clear of 0) is met.

Both controls pass, so the result is interpretable:
- **Decoy:** AUC 0.659 < base v8-in 0.686 + 0.02. The decoy (labels shifted 3.5 × 3.0 mm) is no better than the base model it started from.
  - Caveat: the shift left the decoy only 307 target-pool tiles with 11 ink tiles (the real students had 1168 with 183 ink). So it mostly checks that the anchors alone do not lift T1. It does not fully show that the target labels carry the gain.
- **Depth order:** the students on forward-ordered layers score 0.648 (seeds 0.656 / 0.641 / 0.639), well below 0.765 on reverse-ordered layers.

## All numbers (d = 0)

| Map | AUC | 95 % CI |
|---|---|---|
| Student (mean of 3 seeds) | 0.765 | 0.745-0.822 |
| seed 0 / 1 / 2 | 0.771 / 0.757 / 0.761 | |
| teach4 (v8-in rev, Reader v3, Reader v2, Hecate) | 0.769 | 0.744-0.821 |
| Reader v3 (best single MIT model) | 0.742 | 0.709-0.819 |
| Hecate + Reader v3 | 0.780 | 0.755-0.833 |
| Student + teach4 (equal weight) | **0.796** | 0.772-0.837 |
| v8-in reverse (student initialisation) | 0.686 | 0.659-0.738 |
| Decoy | 0.659 | 0.615-0.698 |

| Difference | Value | 95 % CI |
|---|---|---|
| **D1: student - teach4** | -0.004 | -0.032 to +0.033 |
| student - Reader v3 | +0.023 | -0.033 to +0.070 |
| student - (Hecate + Reader v3) | -0.015 | -0.049 to +0.024 |
| (student + teach4) - teach4 | **+0.027** | **+0.005 to +0.041** |
| decoy - v8-in | -0.027 | -0.086 to -0.001 |

Copy check: the Pearson correlation of the student and teach4 z maps over 7.3 M valid render66 pixels is 0.849. The per-teacher values are v8-in 0.732, Reader v3 0.785, Reader v2 0.785 and Hecate 0.673. The student is close to its teachers, but it is not a copy of any one of them.

## Same measures at d = +1 (one layer deeper, secondary)

| | d = +1 | 95 % CI |
|---|---|---|
| Student / teach4 | 0.765 / 0.767 | |
| student - teach4 | -0.002 | -0.035 to +0.032 |
| student - Reader v3 | +0.026 | -0.029 to +0.073 |
| student - (Hecate + Reader v3) | -0.010 | -0.050 to +0.031 |
| (student + teach4) - teach4 | +0.029 | +0.005 to +0.042 |

The d = +1 numbers tell the same story as d = 0.

## What this does and does not show

- Adaptation lifted the student from its v8-in start (0.686) to the level of the four-teacher ensemble (0.765 vs 0.769), in one ~35 min run per seed on one RTX 3090. It did this without any T1 label. As a single model, it scores above every single MIT teacher on T1, though not significantly.
- It did not beat the ensemble that made its labels. That is the primary, pre-registered question, and the answer on T1 is no.
- Adding the student to the ensemble as a fifth voter at equal weight raised AUC by +0.027, and the CI is clear of 0. This is a pre-registered secondary measure, not the primary one. It comes from one target, and it is not a claim about other scrolls.
- T1 is one scroll and 18 letters. No cross-target claim is made (see PREREG, "Later targets").

## Run notes

- Students were trained 8 Nov, 14:53-17:53, on the local RTX 3090 (peak 8.6 GB).
- Seed 0 was interrupted twice by another session's job and resumed from its checkpoint (Amendments 4-5). All three seeds and the decoy completed 600 steps.
- Student weights: sha256 `90a78e4e…` (s0), `ab58dad9…` (s1), `f4bf5db5…` (s2), `7f3e7c46…` (decoy). They are derived from v8-in (MIT) and trained on MIT-teacher pseudo-labels and the organisers' 1447 labels. They are not released yet.
