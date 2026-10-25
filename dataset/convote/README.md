# Convote Example Data

This directory contains only the preprocessed example inputs, not the original
speech transcripts or the full raw data distribution.

| File | Contents |
| --- | --- |
| `convote.txt` | 219 nodes, 521 undirected ties, 1,042 directed propagation arcs |
| `innate_uniform.txt` | Fixed uniform `[0,1]` opinions, generation seed 219 |
| `innate_exponential.txt` | Fixed normalized exponential-style opinions, generation seed 521 |
| `lt_weights.txt` | Random incoming-normalized propagation weights, generation seed 438 |

The reference graph is derived from the Cornell Convote v1.1 speaker-reference
data. Unordered endpoint pairs are deduplicated, self-loops are removed, nodes
are sorted and renumbered from zero, and each undirected tie is emitted in both
directions. Directed edge lines are lexicographically sorted. In `convote.txt`,
the first comment's edge count (521) is the undirected count; the following
`directed_edges` comment gives the actual propagation arc count (1,042).

Innate opinions and LT weights are generated experimental inputs, not original
Convote annotations. Exponential-style opinions are sampled as
`1 - log(1-U)` and divided by the maximum sampled value. LT weights are sampled
uniformly and normalized per destination so incoming weights sum to at most
one. Their input format is `u v weight`; use them via `--p-file` with LT.

Original dataset source:
<https://www.cs.cornell.edu/home/llee/data/convote.html>.
Citation supplied by the original dataset README:

Matt Thomas, Bo Pang, and Lillian Lee. **Get out the vote: Determining support
or opposition from Congressional floor-debate transcripts.** Proceedings of
EMNLP, pages 327-335, 2006.

The project packaging does not relicense the original dataset.
