# Plan006 review — approved

Commit `7f63f0b1ebaac1e443defb09ab7e3ad6ccc6b608` preserves the final Darwin lifetime peak before child reaping. The reviewer confirmed the installed SDK ABI through a compiled C probe, inspected sampler/launcher/validator/test changes, and independently passed all115 Python tests with ResourceWarnings treated as errors.

A real8MiB child remained waitable through terminal sampling, reported current physical footprint0 with a positive final lifetime peak13,418,760B, and became unavailable to rusage after reap. A separate fresh monitored launcher receipt passed the terminal-policy validator. Live-zero, invalid identity, missing terminal peak, over-budget results and sampling failures remain failures. Historical receipts retain their original legacy policy; new model cohorts require terminal evidence.

One review round fixed persistent exit-observer errors leaving an owned child alive and ensured a reaped-status mismatch records the actual exit code, rejects the terminal observation, and remains a valid failed receipt. No successful sampling path reaps before reading terminal rusage.

Evidence: `.build/flash/runs/reviewer-terminal-lifecycle/`, `reviewer-terminal-launch/`, `sampler-abi-review/layout.json`, and the preserved failing `reviewer-capture-final/` cohort. Fresh Plan005 and Plan004 model cohorts are being rerun after acceptance. No native code/budget change, merge or push.
