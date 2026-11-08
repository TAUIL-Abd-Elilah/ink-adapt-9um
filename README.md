# ink-adapt-9um

Label-free adaptation of a 9 um ink model to a target scroll, with MIT-licensed teachers only, tested on held-out known text.

**Status, 8 Nov 2026: first target (PHerc. 0343) scored. Verdict: "no better than the ensemble".** See `results/T1_RESULT.md`. The student (mean of 3 seeds) scores AUC 0.765 against 0.769 for the four-teacher mean that made its labels (D1 = -0.004, 95 % CI -0.032 to +0.033). Both controls pass. As a fifth voter at equal weight, it lifts the ensemble to 0.796 (+0.027, CI +0.005 to +0.041); that is a pre-registered secondary measure on one scroll. `PREREG.md` (with its amendments) fixed the question, recipe, measures and verdict wording before any student was trained or scored. `FREEZE.txt` holds the original hashes, and each amendment lists the new ones.

## Idea

The PHerc. 0306B First Letters award used a v8-in fine-tune by Warner and Bodill (A-hn). It was trained on machine-consensus labels from other surfaces of the same scroll, plus PHerc. 1447 human labels, with a hard-negative penalty. Its teachers include a non-commercial model, so its weights are non-commercial.

This repo runs the same recipe with MIT teachers only and no labels from the target scroll. It asks one question: is the student better than simply averaging its teachers, measured on text the team already knows?

## First target: PHerc. 0343

- **Known answer:** Erwin Nieuwlaar's 18 letter boxes from the awarded image.
- **Adaptation pool:** his two released meshes, w047 and w048, minus a 3-D zone 2.9 mm around every letter (CT zeroed within 2.2 mm). That leaves about 4.0 cm2.
- **Teachers:** v8-in, Reader v3, Reader v2 and Hecate 9.6 um.

Scores on the letters (`results/T1_BASELINE.md`, `results/T1_RESULT.md`; AUC of in-box against same-row pixels):

| Model or ensemble | AUC |
|---|---|
| Reader v3 | 0.742 |
| Hecate | 0.731 |
| Reader v2 | 0.706 |
| v8-in | 0.686 |
| Four-teacher mean (the bar) | 0.769 |
| Hecate + Reader v3 | 0.780 |
| **Student (this repo, 3 seeds)** | **0.765** |
| **Student + four-teacher mean** | **0.796** |

## Layout

| Path | What |
|---|---|
| `code/prep_pool.py` | Exclusion zone: 3-D distance to the held-out region; CT zeroed under 2.2 mm |
| `code/build_labels.py` | MIT-consensus pseudo-labels with A-hn's rule; writes packs in their format |
| `code/build_anchor.py` | PHerc. 1447 human-label packs (A-hn's `src_1447`) |
| `code/ahn/` | A-hn's trainer and library (MIT, Warner and Bodill), one path change; see `code/ahn/NOTICE.md` |
| `code/eval_student.py` | Student maps with v8-in's own `predict.py` |
| `code/score_t1.py` | The pre-registered score, with a bootstrap over letters |
| `code/e3_t1_gpu.sh` | The T1 GPU job (teachers, labels, 3 students and a decoy, evaluation) |
| `configs/` | Student configs (= A-hn `hn4_s0.json`, seeds 0-2, decoy) |

Renders use the author's Python stand-in for `vc_render_tifxyz` and wrappers for each model; their hashes are in `FREEZE.txt`.

## Credits

- Liam Warner and Nicola Bodill (A-hn recipe and code, MIT).
- Youssef Nader (v8-in, MIT; PHerc. 1447 labels, CC BY-NC 4.0).
- KLAVIS (Reader v3, MIT); Reader v2 (MIT).
- The Vesuvius Challenge team (Hecate, MIT; scans).
- Erwin Nieuwlaar (PHerc. 0343 meshes and letter boxes).
- jeff-j-chen, whose notes on failed cross-scroll ink transfer motivate per-scroll adaptation.

## License

Code MIT. Results and any released weights: CC BY-NC 4.0, because the PHerc. 1447 labels and the scans carry that licence. Third-party code: `code/ahn/LICENSE_ahn`.

Author: TAUIL Abd Elilah. Developed with AI assistance under my direction.
