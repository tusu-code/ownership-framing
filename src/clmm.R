# Ordinal robustness check (added 6 November 2026 in response to review; exploratory).
#   Rscript src/clmm.R   -> extend/out/clmm.json
# Cumulative-link mixed models (logit) of Claude's review-form rating (0, 2, ..., 10) on the author ask
# (author = 1, reviewer = 0), random intercept per paper; with three wordings, wording is a fixed effect.
# Input: extend/out/clmm_input.csv written by src/review_extra.py.
suppressPackageStartupMessages(library(ordinal))
d <- read.csv("extend/out/clmm_input.csv")
d$rating <- factor(d$rating, levels = sort(unique(d$rating)), ordered = TRUE)
d$author <- as.numeric(d$who == "author")
d$pid <- factor(d$pid)
res <- list()
for (s in unique(d$study)) {
  x <- droplevels(d[d$study == s, ])
  x$rating <- factor(as.character(x$rating), levels = sort(as.numeric(unique(as.character(x$rating)))), ordered = TRUE)
  f <- if (length(unique(x$family)) > 1) rating ~ author + factor(family) + (1 | pid) else rating ~ author + (1 | pid)
  m <- clmm(f, data = x, link = "logit", nAGQ = 10)
  b <- coef(summary(m))["author", ]
  res[[s]] <- list(n = nrow(x), papers = nlevels(x$pid), log_or = unname(b["Estimate"]), se = unname(b["Std. Error"]),
                   or = exp(unname(b["Estimate"])), lo = exp(unname(b["Estimate"] - 1.96 * b["Std. Error"])),
                   hi = exp(unname(b["Estimate"] + 1.96 * b["Std. Error"])), p = unname(b["Pr(>|z|)"]))
  cat(sprintf("%-14s n=%4d papers=%3d OR=%.2f [%.2f, %.2f] p=%.2g\n", s, nrow(x), nlevels(x$pid),
              res[[s]]$or, res[[s]]$lo, res[[s]]$hi, res[[s]]$p))
}
writeLines(jsonlite::toJSON(res, auto_unbox = TRUE, digits = 8, pretty = TRUE), "extend/out/clmm.json")
