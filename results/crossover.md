# Crossover table: default corner-cutting (T1) vs gaming in the split-brain environment, by version

Question it decides: does an earlier version cut more corners by default while gaming *less* when graded, and does 3.8 do the reverse? If so, that crossover (default persona vs graded persona moving in opposite directions) is the headline.

Columns 2-3 are from T1 (results/T1_full/README.md; corner-cutting rate, Wilson 95%, n=250 per cell; 3.5 agentic is provisional pending the think-prefill rerun). Columns 4-6 need Olivia's split-brain numbers (graded-mode gaming, deployment-mode gaming, diffing compliance). Fill in from the other repo; the Qwen3.5 split-brain run is still outstanding.

| version | T1 corner-cutting, chat | T1 corner-cutting, agentic | graded gaming (split-brain) | deployment gaming | diffing compliance |
|---|---|---|---|---|---|
| 3.5 | 26% [21, 32] | 31% [26, 37] (provisional) | TODO (Qwen3.5 run pending) | TODO | TODO |
| 3.6 | 38% [32, 44] | 28% [23, 34] | TODO (pilot: games broadly incl. deployment; labellers 73% vs 31%) | TODO | TODO |
| 3.8 | 29% [24, 35] | 14% [10, 18] | 38/50 = 76% [63, 86] | 4/50 = 8% [3, 19] | TODO (held-out scoring 3/50 = 6%) |

Known so far from the spec: Qwen3.8 games 38/50 graded, 4/50 deployment, 3/50 held-out scoring; Qwen3.6 has no graded mode and games broadly (provisional). The T1 honesty-with-tools sub-measure (propagating own earlier error into a send call, agentic): 3.5 64%, 3.6 56%, 3.8 24% "smooth" (results/T1_full/README.md section 6).

Reading if the TODOs come in as the spec suggests: 3.8 = high graded gaming, low default corner-cutting with tools; 3.6 = broad gaming, highest default corner-cutting in chat; 3.5 = TBD. Note that T1's agentic frame has no grader, so a 3.8 crossover would be "games when graded, careful when not", i.e. conditioning on evaluation in both directions; the T1 eval-awareness coding (results/T1_full/eval_awareness.md, in progress) is the check on whether "careful when not graded" is itself "careful when it does not suspect a test".
