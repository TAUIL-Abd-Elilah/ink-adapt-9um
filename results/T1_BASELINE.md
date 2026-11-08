# E3 T1 (PHerc0343) baseline, 7 Nov 2026

Target: Nieuwlaar's 18 awarded letter boxes on render66 (crop u400-5800 v400-2300 of his concat_w047-w048 mesh).
Scorer: `_fl/route24/score_letters_0343.py`, reading-side (reverse) maps, AUC of in-box pixels vs same-row outside pixels.
Score files: `letter_scores_d+0.json`, `letter_scores_d+1.json` (this folder). All on CPU.

## Single MIT models, d+0 (d+1 in brackets)

| Model | AUC |
|---|---|
| Reader v3 (new today) | 0.742 (0.739) |
| Hecate 9.6 um | 0.731 (0.733) |
| Reader v2 | 0.706 |
| d9v4C (ours, distilled) | 0.694 |
| v8-in, reverse order | 0.686 |

## MIT ensembles (mean of z-scored maps), d+0

| Ensemble | AUC |
|---|---|
| Hecate + Reader v3 | **0.780** (d+1 0.775) |
| Hecate + Reader v2 + Reader v3 | 0.772 |
| Hecate + Reader v2 + Reader v3 + v8-in (the 4-teacher label source) | 0.769 |
| Hecate + Reader v2 (previous best MIT) | 0.749 |

For reference (NC, not usable as a teacher): TRACE v3B 0.750; Hecate + TRACE 0.778.

**The MIT pair Hecate + Reader v3 matches the best NC combination on this target.** The bar for an adapted student is the teacher ensemble, not the best single model. Which ensemble is "the" comparator must be fixed in PREREG before any student is scored. Choosing the 2-model pair from this table would be tuning on the target.

## Adaptation pool for T1

- Erwin's repo releases only w047 and w048 (github.com/Nieuwlaar/pherc343-first-letters, outputs/). Blob hashes match our copies (x.tif e4f56e6b...).
- 3-D exclusion (A-hn rule 2.0 + 0.2 + 0.7 mm = 336 vox from any node inside a letter box) leaves 1.98 cm2 of w047 and 2.03 cm2 of w048: **4.0 cm2 pool**. w047 and w048 overlap partly (21.5 % of w047 lies within 0.3 pitch of w048).
- The open-data bucket has no PHerc0343 segments; PHerc0343P is a different object.
- Pool renders: `C:/VesuviusTemp/nov_ink/t1_343/renders/w04{7,8}.zarr` (66 layers, CPU).

## Second eval source found

A-hn's `v8306lib.src_0343` names jeff-j-chen's 343 stroke tracings (MIT, commit 5f6f634), registered to the concat render. These would give a pixel-level T1 check besides the boxes. They must never enter a T1 student's training.
