# Language models rate a manuscript by who is asking

Code for the study of how a user's stated relation to a manuscript (author, colleague, reviewer, none) changes the verdict that frontier language models give on the same paper. The main study sent 342 ICLR 2026 submissions, disguised as unpublished work, to five models in independent conversations that differed in one sentence about the user. Three preregistered studies on new arXiv papers then tested the central contrasts and two requests a user can make:

| Study | Registration | Folder |
|---|---|---|
| Confirmatory test of the four central contrasts (Claude Sonnet 4.6, 160 papers, 3 wordings) | https://osf.io/jcaqw | `confirm/` |
| A request to review instead of an opinion (Claude Sonnet 4.6 and Gemini 4 Argon, 240 papers) | https://osf.io/sz6n4 | `mitig/` |
| A request to judge the paper as someone else's (Claude Sonnet 4.6, 200 papers) | https://osf.io/65qph | `detach/` |
| Added exploratory studies: papers outside machine learning; user-side requests | not registered | `extend/` |

Author: anonymous (identity withheld for peer review).

## Layout

| Folder | Contents |
|---|---|
| `src/` | Main study: manuscript preparation and disguise, the asks, model calls, second-turn ratings, blind readers, recognition, tone and reasoning checks (collection, as run); statistics, the number files quoted in the paper, tables and figures for all studies (analysis) |
| `confirm/`, `mitig/`, `detach/` | Each registered study: `PREREGISTRATION.md`, `FROZEN.sha256`, the frozen scripts in `src/` and the sample list and material log in `data/` |
| `extend/` | The added studies: sampling frame, materials, asks and runner |
| `paper/` | Source of the design figure and the icons used in the figures (licences in `paper/icons/LICENSES.md`) |

## Checking the frozen files

The scripts each registration froze are included unchanged. From the repository root,

```
sha256sum -c confirm/FROZEN.sha256
sha256sum -c mitig/FROZEN.sha256
sha256sum -c detach/FROZEN.sha256
```

reports every script and list as OK, except three kinds of entry. The `*/data/manuscripts.jsonl` files hold third-party manuscript text, are not redistributed and are rebuilt from arXiv by the materials scripts below. `confirm/src/frame.py` and `confirm/src/materials.py` differ from their registered copies only in the contact address in the download step, which is masked here for anonymous review.

## Data

- Main study: public ICLR 2026 materials from OpenReview, as redistributed in the Hugging Face datasets `weathon/iclr_2026` (full texts) and `JerMa88/ICLR_Peer_Reviews_2026` (reviews and decisions). Neither states a licence, so this repository contains no manuscript text, original or disguised. Place the inputs as `data/raw/papers/<paper_id>.txt` (full text, references and appendix removed) and `data/raw/meta_396.csv` (paper_id, authors, n_reviews, human_mean, ratings, primary_area, decision, title).
- Registered and added studies: public arXiv papers, listed by arXiv number in `*/data/sample_ids.csv` and `*/data/material_log.csv`; `confirm/data/frame.csv` is the random sampling frame shared by the three registered studies.
- Model answers, neutralized rewrites, judge outputs and second-turn ratings are released separately with the paper (OSF). The main study's answers are keyed by pseudonymous paper codes; the disguised manuscripts and the mapping to OpenReview entries are available under controlled access on request.

## Setup

```
pip install -r requirements.txt
cp config.example.yaml config.yaml      # set base_url to your OpenAI-compatible endpoint
cp .env.example .env                    # fill in API keys; .env is ignored by git
```

All models are called through an OpenAI-compatible endpoint (Claude through the Anthropic messages API). Every call is a new conversation; `python src/run.py smoke --model mock` tests the pipeline without any API call. The registered studies use the main study's client (`src/run.py`), `config.yaml` and `.env`, and are run from the repository root. Set `API_GATEWAY_PREFIX` if your result files mix records from more than one endpoint. The cumulative-link mixed models use R 4.6 with the `ordinal` package.

## Main study

Run from the repository root. Every step skips finished work, so it can be stopped and restarted.

| Step | Command |
|---|---|
| Clean manuscripts, draw the sample | `python src/prepare.py` |
| Disguise (rename, retitle, paraphrase) | `python src/disguise.py run --sample sample_full.csv` |
| Asks in each format (`--cond` selects asks, e.g. `open_author`, `struct_reviewer`, `fixed_author`, `open_filler`) | `python src/run_p2.py review --full --model claude --cond open_author --cond open_reviewer` |
| Neutralize free-text answers and blind-read them | `python src/run_p3_judge.py run --struct` |
| Second-turn rating by the model itself | `python src/followup.py --model claude --sample sample_full.csv --cond open_author --cond open_reviewer` |
| Independent reader (MiniMax), neutralized and raw text | `python src/judge3.py neutral raw` |
| Second reader on a subsample (Claude Haiku 4.5) | `python src/judge2.py` |
| Recognition test | `python src/recognize.py --model gemini` |
| Tone test | `python src/tone_swap.py rewrite && python src/tone_swap.py judge` |
| DeepSeek reasoning check | `python src/reasoning_check.py` |
| Statistics | `python src/stats_full.py` |

The shell scripts in `src/` (`rev_chain.sh`, `rev_extra.sh`, `rev2_chain.sh` and others) run these steps per model in the order used in the study. The analysis reads `full_study/data` and `full_study/results`; link them to the collection output with `mkdir -p full_study && ln -s ../data full_study/data && ln -s ../results full_study/results`. Early pilot scripts (`analyze.py`, `analyze_p2.py`, `status_probe.py`) and interim checks (`round3_check.py`, `gemgrok_check.py`, `revision_check.py`) are kept for completeness.

## Registered studies

Each follows the same steps; see the docstring of each script and the study's `PREREGISTRATION.md`.

| Step | Confirmatory | Request to review | Judge as someone else's |
|---|---|---|---|
| Sampling frame | `python confirm/src/frame.py` | (shared frame) | (shared frame) |
| Download, convert, select, audit | `confirm/src/materials.py convert`, `select`; `confirm/src/leak_audit.py` | `mitig/src/materials_mitig.py convert`, `select`, `audit` | `detach/src/materials_detach.py convert`, `select`, `audit` |
| Run (smoke test, then sample) | `confirm/src/run_confirm.py run --role sample` | `mitig/src/run_mitig.py run --model claude --role sample` (and `--model gemini`) | `detach/src/run_detach.py run --role sample` |
| Recognition probe | `run_confirm.py recognize --role sample` | `run_mitig.py recognize --model claude --role sample` | `run_detach.py recognize --role sample` |
| Registered analysis | `confirm/src/analyze_confirm.py --role sample` | `mitig/src/analyze_mitig.py --role sample` | `detach/src/analyze_detach.py --role sample` |

`detach/src/dl_ids.sh` and `extend/src/dl2.sh` download first-version arXiv PDFs with a completeness check (the file must start with `%PDF` and end with `%%EOF`).

## Added studies

`python extend/src/frame_extend.py`, then `extend/src/materials_extend.py` and `python extend/src/run_extend.py run --study A --role sample` (papers outside machine learning) or `--study B` (user-side requests).

## Numbers, tables and figures for the paper

```
python src/paper_numbers.py && python src/confirm_numbers.py && python src/mitig_numbers.py \
  && python src/detach_numbers.py && python src/review_extra.py && Rscript src/clmm.R && python src/extend_numbers.py
python src/paper_tables.py && python src/style_tables.py
python src/make_icons.py && python src/paper_figs.py && python src/fig_borderline.py
```

`style_tables.py` adds the colour highlighting to the generated tables (registered results and verdicts); the paper is built with `xelatex si; xelatex main; xelatex si; xelatex main`, as the main text and the Supplementary Information refer to each other.

Every number in the paper is written by these scripts to `paper/numbers*.tex` and quoted from there; none is typed by hand. The number scripts for the registered studies read only the output of the frozen analysis scripts.

## AI use

Claude Opus 5.5 (Anthropic) wrote most of this code under the author's direction; the author reviewed it and is responsible for it. See the paper's Methods.

## Licence

MIT (see `LICENSE`). The icons in `paper/icons` keep their own licences.
