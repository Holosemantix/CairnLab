# Missing raw fixed-pool tail data

The retained Paper1 artifacts contain aggregate q50/q90 candidate-margin and paired-drift summaries plus fixed-pool flip rates. They do not contain sample-level candidate-cost traces, per-anchor maximum cost drift, q10 clean-margin tails, q95/q99 max-drift tails, or pool-level sufficient-event pass flags. Accordingly, the fixed-pool audit reports observed Top1Agree and q50/q90 proxy gaps, and leaves cert_pass_pool/cert_pass_rate fields empty rather than inferring them from summaries.
