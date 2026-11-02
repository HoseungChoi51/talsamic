# LLM descriptor tables for the benchmarks

These were made by `python -m yieldengine.llm.descriptors` with a local LLM (qwen3.8-flash-next, 3 samples each) on 2026-11-02. They aren't confirmed by an engineer (`confirmed_by` is null), which is allowed for benchmarks only.

Use them with `--descriptors engine/descriptors` on `yieldengine.benchmarks.run` or `.campaign`.

Each table holds the median of the samples (`values`), how far the samples disagreed (`spread`), the descriptor list and ranges that the engine fixed (`spec`), and the model's own caveats (`notes`).

Known issue in `direct_arylation.json`: for butyronitrile, the dielectric constant is 20.3 with a spread of 16. The literature value is about 24.8. One sample misread the SMILES as valeronitrile.

Known issues in `buchwald_hartwig.json`:
- The additive table was redone with 3 samples (30 min, 0 issues), because two calls in the first run hit the old 600 s timeout. The medians barely changed.
- `n_o_bond_lability` for the additives looks the least chemically grounded. The C3-H isoxazoles (A01, A12, A19, A20), which should open their ring most easily under base, don't stand out. A12 (5-methylisoxazole) is among the lowest. The column seems to follow "has an electron-withdrawing group or a fused ring" instead.
- `hindrance_near_n`: A02 (4-phenylisoxazole, C3-H next to N) is rated 4, which is too high. For A04, one sample scored the dibenzylamine nitrogen instead of the ring N (spread 4).
- The BTMG and MTBD pKa values vary by about 9–10 between samples, and they look like MeCN values rather than DMSO values. Their order is still right.

Known issues in `suzuki.json`:
- The ligand table has no values for P(Ph)3. The data file names it "P(Ph)3 " with a trailing space, and the descriptor tool didn't match the LLM's trimmed name. Names are now trimmed, and the ligand table will be redone. The redo also gives the LLM each ligand's structure, since the first run took AmPhos for a biaryl phosphine and CataCXium A for a ferrocene bisphosphine.
- `halide_reactivity` is null for the boron reagents in the electrophile column, because it doesn't apply to them. The model then sees the column average, and `ring_electron_poor` (4–5 for all) carries little information.
- For "2d, Bromide" as a nucleophile, the reactivity and polarity values have a spread of 3, because a bromide isn't a nucleophile in this reaction.
