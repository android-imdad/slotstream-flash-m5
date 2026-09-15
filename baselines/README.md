# Benchmark archives

Version control contains the frozen reports, requests, manifests and harnesses.
The large `*/bin/` executable and Metal-library files stay in the local archive;
they are excluded from Git. Their original hashes remain in each archive's
`SHA256SUMS.json`. A fresh clone therefore cannot reproduce an archived executable
run without those matching local binaries. Never replace them with a new build
and describe it as the original reference.

The JANG comparison runner consumes the saved requests and target numbers from
`baseline.json`; its controlled before/after binaries are separately archived
under `.build/flash/runs/`. Full raw Flash run artifacts also remain under that
ignored directory. The `plans/` execution records identify their exact paths and
methods. No remote upload of these large local artifacts is implied by a commit.

Current candidate outcomes are maintained in [JANG Flash findings](../docs/JANG-FINDINGS.md)
and the [plan index](../plans/README.md). A frozen archive's README records the
state at capture time; later progress does not rewrite that archive or its hashes.
