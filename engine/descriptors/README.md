# LLM descriptor tables for the benchmarks

These were made by `python -m yieldengine.llm.descriptors` with a local LLM (qwen3.8-flash-next, 3 samples each) on 2026-11-02. They aren't confirmed by an engineer (`confirmed_by` is null), which is allowed for benchmarks only.

Use them with `--descriptors engine/descriptors` on `yieldengine.benchmarks.run` or `.campaign`.

Each table holds the median of the samples (`values`), how far the samples disagreed (`spread`), the descriptor list and ranges that the engine fixed (`spec`), and the model's own caveats (`notes`).

Known issue in `direct_arylation.json`: for butyronitrile, the dielectric constant is 20.3 with a spread of 16. The literature value is about 24.8. One sample misread the SMILES as valeronitrile.

Known issues in `buchwald_hartwig.json`:
- The additive table rests on one sample, because two calls hit the old 600 s per-call timeout. Its spread is null, meaning unknown. It will be redone with 3 samples and a longer timeout using `--variables additive`.
- `n_o_bond_lability` for the additives looks the least chemically grounded. The C3-H isoxazoles (A01, A12, A19, A20), which should open their ring most easily under base, don't stand out.
- The BTMG and MTBD pKa values vary by about 9–10 between samples, and they look like MeCN values rather than DMSO values. Their order is still right.
