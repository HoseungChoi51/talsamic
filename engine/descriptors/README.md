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
- The ligand table was redone (18 min, 0 issues) after two fixes. The data file names one ligand "P(Ph)3 " with a trailing space, so the first run left it empty. And the first run took AmPhos for a biaryl phosphine and CataCXium A for a ferrocene bisphosphine; the LLM now gets each ligand's structure, and both are classified correctly. The cone angles still run high (AmPhos 210° with a spread of 39, against a realistic ~170–180°; SPhos, dtbpf and XPhos are also at 210–220°). Treat `cone_angle_deg` as loose; `steric_bulk` orders the ligands better.
- `halide_reactivity` is null for the boron reagents in the electrophile column, because it doesn't apply to them. The model then sees the column average, and `ring_electron_poor` (4–5 for all) carries little information.
- For "2d, Bromide" as a nucleophile, the reactivity and polarity values have a spread of 3, because a bromide isn't a nucleophile in this reaction.
