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

## Amendment 2 (8 Nov 2026, 13:35, before any student was trained)

Another session started a small trainer (2.6 GB) while our Hecate teacher step ran. `gpu_guard.py` now waits only for dedicated VRAM under 4 GB, which allows a small foreign job beside ours, as the owner's GPU rule does. It no longer waits for every foreign trainer to finish. `vram_watch.py` is unchanged: it stops our trainer if a foreign trainer runs and VRAM passes 22.5 GB. Nothing computed changes. New hash:

384cee535ad2db94972439031bcb62665c4c9fe35c8efa87d268a32bc0e2b27e *code/gpu_guard.py

## Amendment 3 (8 Nov 2026, 13:45, before any student was trained)

The v8-in teacher step on pool render w047 used 16.9 GB at the wrapper's inference batch of 16. With another session's trainer at 5.8 GB, the card reached 24 GB. Our step was stopped, and the other job was left running. Both pool v8-in maps (w047, w048) are now made by `code/v8in_small.py` at batch size 4, with a 20 GB VRAM cap. It uses the same 24 layers and the same `predict.py` and checkpoint as `v8in_on_render.py`. Tiles are predicted independently in eval mode and averaged on the CPU, so the batch size does not change what is computed. The render66 evaluation maps are unchanged. New hash:

8c765e1d0830f915139239157ef4366182b4179409c1878bbfc4b115c230e6a6 *code/v8in_small.py

## Amendment 4 (8 Nov 2026, 15:05, after 40 steps of student t1_s0 and before any student finished)

At 15:00 our trainer (9.2 GB) and another session's trainer (13.6 GB) brought the card to 23.9 GB, and the other job began spilling into shared memory. The job's `vram_watch.py` never ran: Git Bash's `$!` is an MSYS pid, not a Windows pid, so it exited at once. We killed our trainer by hand. It resumes from `state.pt`, so the run's step count and schedule are unchanged.

Two changes:
- `gpu_guard.py` again waits until no foreign trainer runs, as well as VRAM < 4 GB. Both must hold on two polls in a row, because the other session starts its next run within about 20 s. This reverts the relaxation in Amendment 2, which proved unsafe for training: the other job grows from 3.6 GB to 13.6 GB.
- `vram_watch_win.py` (with `gpu_guard_fn.py`) runs beside the job. It finds our trainer by its command line and kills it if a foreign trainer runs and VRAM passes 21 GB.

Nothing computed changes. New hashes:

037f361d4f5bc53a71d928f8dca2d4a4234a9490e7ade81a22ec3f1c5c149949 *code/gpu_guard.py
0caaff4204f6cf2092e51999d6f16520cbc45d3d24d013fa6e775c7146a2458c *code/vram_watch_win.py
b2c7b78a3f9ed4658b8158ba11fb5cc7237ae7d7877469f2d8ed7a2da7607ec5 *code/gpu_guard_fn.py

## Amendment 5 (8 Nov 2026, 15:30, student t1_s0 at step 30 of 600, none finished)

At 15:19 attempt 2 of t1_s0 also started in the gap between two chained runs of the other session. It was stopped by `vram_watch_win.py` at 24.0 GB. Nothing spilled to shared memory, and the run resumes from step 30. The other session runs its trainings from a chain shell (`bash scratch/queue*.sh` in D:/Competition/wundernn). The First Letters session suggested that `gpu_guard.py` also wait while such a shell is alive, and it now does. Nothing computed changes. New hash:

1afe9b92e0d3ffa38d98095ab6d3f224899fb6e858c74827db3878d55a6991e4 *code/gpu_guard.py

## Amendment 6 (8 Nov 2026, 16:55, two students trained, no student evaluated)

`code/eval_student.py` did not compile. It had the same backslash escaping slip as Amendment 1 (`rstrip('/\')` written as an unterminated string), so the job would have stopped at its evaluation step. Line 13 now reads `rstrip('/').rstrip(chr(92))`, and nothing else changed. All other scripts compile. A CPU smoke run on a 256 × 256 crop of render66 checks the path.

`configs/spec_t1.json` is the scoring spec for the measures above. Its "student_plus_teach4" entry gives the student and teach4 equal weight: `E3T1` is listed four times beside the four teachers, because `score_t1.py` takes a plain mean of z maps. It is fixed before any student map exists.

New hashes:

e4878bdd97d20e5749b5c2f86ec8fd4138e4ca9c4287cdec0d4d96b7d4f790d7 *code/eval_student.py
0af1d51dfa95c04f68139c4955f52475b1d7f02bf14bb4a249f6bb51f323c8ee *configs/spec_t1.json
