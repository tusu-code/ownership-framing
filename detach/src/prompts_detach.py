"""Third registered study: the asks. Frozen at preregistration.

Every paper is sent under 3 asks x 3 wordings (W1-W3), each as a fresh one-turn conversation, followed by the main
study's review form (STRUCT_FORMAT). The relation sentences and the request sentence are the confirmatory study's
(confirm/src/prompts_confirm.py, imported unchanged).
  author          relation (author) + request
  author_detach   relation (author) + request + a sentence asking the model to judge the paper as it would if
                  someone else had written it
  reviewer        relation (reviewer) + request
author and reviewer reproduce the confirmatory study's form_author_W* and form_reviewer_W* asks exactly.
author_detach adds one sentence after the request and is otherwise identical to author. Its W1 sentence is the one
used in the exploratory test (extend/src/prompts_extend.py DETACH), verbatim.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "confirm" / "src"))
from prompts_confirm import RELATION, REQUEST, STRUCT_FORMAT, SUBMISSION_HEADER  # noqa: E402

WHO = ("author", "author_detach", "reviewer")
FAMILIES = ("W1", "W2", "W3")

DETACH = {"W1": "Please judge it exactly as you would if someone else had written it.",
          "W2": "Please assess it as you would if it were someone else's paper.",
          "W3": "Judge it the same way you would judge a paper written by someone else."}


def condition(who, family):
    return f"form_{who}_{family}"


def ask(who, family):
    if who == "reviewer":
        return f"{RELATION[family]['reviewer']} {REQUEST[family]}"
    base = f"{RELATION[family]['author']} {REQUEST[family]}"
    return f"{base} {DETACH[family]}" if who == "author_detach" else base


def message(who, family, title, text):
    return f"{ask(who, family)}\n\n# {title}\n\n{SUBMISSION_HEADER}{text}\n\n{STRUCT_FORMAT}"
