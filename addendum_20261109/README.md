# Addendum (9 November 2026): consequences and real requests

Data for the two exploratory analyses added to the manuscript on 8 November 2026 (Methods, "Consequences and real requests"; Supplementary Methods 6). They are produced by `src/human_scale.py` and `wildchat/src/{candidates,classify,summarize}.py` in the code repository.

## consequences/
- `claude_form_verdicts_by_paper.csv`: main study, Claude Sonnet 4.6 review form; for each paper (pseudonymous code, as in the main data release) the recommendation under the author and the reviewer ask (1 = accept), the mean ICLR reviewer rating, ICLR's decision (1 = accepted), borderline status and the number of reviews.
- `iclr2026_decided_submissions.csv`: all ICLR 2026 submissions with a public decision (mean reviewer rating, number of reviews, decision), in random order and without identifiers; used for the illustrative acceptance curve.
- `human_scale.json`: the estimates reported in the paper.

## wildchat/
Source: WildChat-1M (allenai/WildChat-1M, ODC-BY; Zhao et al., ICLR 2024). No user text is included; each request is identified by WildChat's `conversation_hash`, from which the text can be retrieved in the public dataset. Users are given pseudonymous codes (U00001...) in place of WildChat's hashed IP addresses.
- `candidates_meta.csv`: the 1,867 first user turns selected by the keyword filter (language, model, date, length, user code).
- `labels_deepseek_v4.1_flash.jsonl`, `labels_claude_haiku_4.5.jsonl`: labels from the two classifiers on all candidates; `labels_gpt_6.1_sol.jsonl`: the third classifier on the candidates either of the first two called an assessment. Fields: `assess`, `relation` (own, reviewer, other, none), `role_play`, `model_id` (the classifiers' quoted evidence is withheld because it reproduces user text).
- `final_labels_majority.csv`: requests labelled an assessment by at least two classifiers, with the majority relation ("unresolved" if none).
- `summary.json`: counts, shares, agreement and the per-user sensitivity analysis reported in the paper.
