# Relevance Grading Rubric

Graded relevance for `expected.graded_chunks`. Every chunk in a case's pool gets
one of these grades. Chunks not in the pool are treated as grade 0.

| Grade | Meaning |
|---|---|
| **3** | Answers the query completely. You could respond from this chunk alone. |
| **2** | Carries a necessary part of the answer, but is insufficient by itself. |
| **1** | On-topic context or a pointer. Contributes nothing to the answer. |
| **0** | Irrelevant. |

## The binary threshold

`recall_at_k` and `mrr` treat **grade >= 2** as relevant. This falls out of the
rubric rather than being asserted separately: grades 3 and 2 contribute to the
answer, grade 1 does not. `nDCG@k` uses the full 0-3 scale.

## Worked examples

**Query:** "How many layers are in the encoder stack of the Transformer?"

- **3** — a chunk containing "The encoder is composed of a stack of N = 6
  identical layers." Complete answer, no further context needed.
- **2** — a chunk describing what each encoder layer contains (self-attention
  plus feed-forward sub-layers) without stating the count. Necessary to
  understand the answer, insufficient alone.
- **1** — the architecture overview naming an encoder and a decoder without
  detailing either. On topic, contributes nothing.
- **0** — the section on training hardware and schedule.

**Query:** "What problem does meta-prompting solve that few-shot prompting does not?"

- **3** — a chunk stating the contrast directly.
- **2** — a chunk describing one side of the contrast only (meta-prompting's
  mechanism, or few-shot's limitation) — a multi-hop case, where two grade-2
  chunks together make the answer and neither is sufficient alone.
- **1** — a chunk defining prompting generally.
- **0** — the related-work section on unrelated techniques.

## Pooling and what "exhaustive" means

Grading all 148 chunks against all cases is not tractable by hand, so grades are
assigned over a **pool**: the union of the top 15 results across the retrieval
configurations under comparison. Anything outside the pool scores 0. This is
standard TREC practice.

**Known bias:** a future retriever that surfaces a genuinely relevant chunk no
pooled configuration ever ranked is scored as though it retrieved garbage. At
depth 15 over a 148-chunk corpus (~10%) the effect is small, but it is real, and
it means nDCG understates a retriever that finds something new. Deepen the pool
before trusting a close result.

**Exhaustiveness matters most for `recall_at_k`,** whose fractional score assumes
a case's label set is complete. A partial label set makes a good retriever look
worse than it is.

## Grading consistently over time

With a single grader, inter-rater reliability becomes **intra**-rater: re-grade a
held-out sample weeks later and compare against the original. Disagreement means
the rubric wording needs tightening, not that the earlier grade was wrong. Do this
before scaling the dataset, and keep the re-graded sample — it is the human label
set that judge validation will need later.
