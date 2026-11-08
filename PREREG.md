# PREREG: label-free target-scroll ink adaptation at 9 um, MIT models only

Written 7 Nov 2026, before any student model is trained. Frozen by `FREEZE.txt` (sha256 of this file and of the code and configs it names) in a public commit pushed before the first student run. Changes after the freeze go at the end of this file as dated amendments, before the runs they affect.

## Question

The PHerc. 0306B First Letters award used a v8-in fine-tune (A-hn, Warner & Bodill). It was trained on machine-consensus pseudo-labels from other surfaces of the same scroll plus PHerc. 1447 human labels, with a hard-negative penalty. Its teachers include a non-commercial model, and its weights are non-commercial.

**Does the same recipe, run with MIT teachers only and no labels from the target scroll, make the target's held-out known text more separable than the teachers' own ensemble?**

## T1: PHerc. 0343 (this pre-registration)

**Known answer (evaluation only).** The 18 letter boxes Erwin Nieuwlaar placed for the awarded PHerc. 0343 image, on our 66-layer render of his concatenated mesh `concat_w047-w048_R5B2_z9500-11000` (crop u400-5800, v400-2300; `render66`).

**Adaptation pool (no labels).** His two released meshes, w047 and w048 (github.com/Nieuwlaar/pherc343-first-letters, `outputs/`), rendered with 66 layers on volume `PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr`. A-hn's exclusion rule applies:
- CT is zeroed on all layers wherever a pixel's 3-D distance to any letter-box node is under 2.2 mm (`prep_pool.py`).
- Only 64-px tiles lying entirely at 2.9 mm or more are used.
- This leaves about 4.0 cm2 (1.98 cm2 on w047, 2.03 cm2 on w048).

**Teachers (labels only, MIT).** v8-in (reverse order), Reader v3, Reader v2 and Hecate 9.6 um, all at d = 0 on the reading side of the zeroed pool renders.
- This is A-hn's teacher set with its one non-commercial member (TRACE v3B) replaced by Hecate.
- ink_9um is not a teacher. It was the weakest model on our earlier held-out human-label test (mean AUC 0.79 against 0.86 for Reader v2, `_fl/dist9v2`), and A-hn did not use it.
- **Disclosure:** the single-model T1 scores in `results/T1_BASELINE.md` were computed before this teacher set was fixed.

**Pseudo-labels.** A-hn's rule (`build_labels.py`):
- Each teacher is z-scored over valid pool pixels; the four are averaged and re-z-scored.
- Ink: fused z ≥ 2.5 in 8-connected blobs of 0.3-6 mm².
- Background: fused z ≤ 0.3 and at least 20 px from ink.
- Everything else is ignored.

**Human half of each batch.** PHerc. 1447 w058 and w060 hand-refined labels (Youssef Nader, CC BY-NC 4.0), as in A-hn. No PHerc. 0343 or 0343P label is used anywhere.

**Student.** A-hn's trainer (`ahn/train_cloud_hn.py`, MIT, one path change):
- Base v8-in (sha256 3b94548d...).
- Config `configs/t1_s{0,1,2}.json` = their `hn4_s0.json`: 600 steps, micro 4 (2 human + 2 pool tiles), accumulation 4.
- Pool weight 0.5; AdamW at 2e-5 cosine to 2e-6 with 5 % warm-up; weight decay 1e-6; label smoothing 0.25; online hard-negative gain 4 on both halves.
- Two changes for a 24 GB card: block gradient checkpointing (same maths) and a VRAM cap of 18 GB.
- Seeds 0, 1 and 2. **The student map is the mean of the three seeds' probability maps.**

**Decoy.** Seed 0 trained on the pool labels shifted by (3.5, 3.0) mm (`configs/t1_decoy.json`).

## Measures (T1)

All measures are computed with `score_t1.py` on render66 at d = 0, reading side:
- Per map: z-score over valid pixels, then the AUC of in-box pixels against same-row out-of-box pixels within the text span. This is the statistic of `score_letters_0343.py`.
- Uncertainty: a bootstrap over the 18 letters (1000 resamples, paired across maps, each out-of-box pixel assigned to the nearest box of its row); 95 % percentile CI.

| | Definition |
|---|---|
| **Primary** | D1 = AUC(student) - AUC(teach4), where teach4 = mean z of the four teachers on render66 (baseline 0.769) |
| Secondary | student - Reader v3 (best single MIT model on T1, 0.742); student - (Hecate + Reader v3) (0.780, the hardest MIT pair, chosen after seeing T1); (student + teach4) - teach4; the same at d = +1 |
| Copy check | Pearson correlation of the student and teach4 z maps over valid render66 pixels (reported) |

**Verdicts:**
- **"adaptation helps on T1"**: D1 ≥ +0.02 and the CI lower bound > 0.
- **"worse than the ensemble"**: D1 ≤ -0.02 and the CI upper bound < 0.
- Otherwise **"no better than the ensemble"**.

**Controls; if either fails, T1 is "not interpretable" whatever D1 is:**
- Decoy: AUC(decoy) < AUC(base v8-in, reverse) + 0.02 (base 0.686).
- Depth order: the student run on forward-ordered layers scores below the student on reverse-ordered layers.

All numbers are reported whatever the verdict, including the per-seed AUCs.

## Later targets

T0 (PHerc. 0306B), T2 (PHerc. 0841) and T3 (PHerc. 1447) are added by amendment, each before its own student runs, with its own data, exclusion and comparator. The overall claim across targets is fixed in that amendment. No cross-target claim will be made from T1 alone.

## Amendment 1 (8 Nov 2026, before any student was trained)

`code/gpu_guard.py` and `code/vram_watch.py` detected another session's trainer only by the file name `train.py`. The trainer running on this machine at 13:01 is `train_arch.py`. Both now treat any `*train*.py` script, except our own `train_cloud_hn.py`, as foreign. These two scripts only decide when a GPU step may start, or stop our trainer to avoid a shared-GPU freeze (it resumes from its saved state). They change nothing that is computed. New hashes:

c3aad20ccbd532bc4c01b06cc030a5fb1340b56ba32611faf2ae27589dfb3d4c *code/gpu_guard.py
8a96c719968ebe7c688c2e2b50d4ffac5efee11c62482ac61936150037ba07ab *code/vram_watch.py
