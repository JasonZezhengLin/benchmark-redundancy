# Data provenance

All model-by-benchmark scores in this study are real public leaderboard data.
No score is hand-entered or synthetic. Two source datasets were retrieved and
one merged matrix was derived from them.

## Retrieval
Retrieval date: 2026-07-19 (single snapshot).
Method: direct download of the Parquet export served by the Hugging Face
dataset viewer conversion branch (refs/convert/parquet).

## Source 1: Open LLM Leaderboard v2 (current)
- Dataset: open-llm-leaderboard/contents
- Landing page: https://huggingface.co/datasets/open-llm-leaderboard/contents
- Parquet: https://huggingface.co/datasets/open-llm-leaderboard/contents/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet
- Local file: ollb_v2.parquet
- Rows: 4576 entries (4497 unique models).
- Benchmarks (normalized 0-100 for ranking; paired "* Raw" columns give 0-1 accuracies for the binomial floor): IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO.
- Task documentation: https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about
- Score normalization: https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization

## Source 2: Open LLM Leaderboard v1 (archived)
- Dataset: open-llm-leaderboard-old/contents
- Landing page: https://huggingface.co/datasets/open-llm-leaderboard-old/contents
- Parquet: https://huggingface.co/datasets/open-llm-leaderboard-old/contents/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet
- Local file: ollb_v1.parquet
- Rows: 7260 entries (6812 with all six benchmarks scored).
- Benchmarks (reported accuracies 0-100): ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K.
- Archive documentation: https://huggingface.co/docs/leaderboards/open_llm_leaderboard/archive

## Derived matrix 3: merged 12-benchmark matrix
- Inner join of Source 1 and Source 2 on the exact model identifier (fullname), keeping models scored on all twelve benchmarks.
- Models in merged matrix: 397. No score altered; each cell is the value reported by the corresponding leaderboard.

## Item counts for the binomial noise floor (FDR)
Standard public test-set sizes; used only to derive se = sqrt(p(1-p)/n). Sources: benchmark papers and the Eleuther lm-evaluation-harness leaderboard task group.
- IFEval: 541 (arXiv:2311.07911); BBH: 6511 (arXiv:2210.09261); MATH Lvl 5: 1324 (arXiv:2103.03874)
- GPQA: 1192 (arXiv:2311.12022); MuSR: 756 (arXiv:2310.16049); MMLU-PRO: 12032 (arXiv:2406.01574)
- ARC-Challenge: 1172 (arXiv:1803.05457); HellaSwag: 10042 (arXiv:1905.07830); MMLU: 14042 (arXiv:2009.03300)
- TruthfulQA MC2: 817 (arXiv:2109.07958); Winogrande: 1267 (arXiv:1907.10641); GSM8K: 1319 (arXiv:2110.14168)
