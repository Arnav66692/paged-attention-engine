# Contributing

Open a focused pull request against `main`. Explain the behavior changed and
include the checks you ran. Follow the setup and verification commands in the
README. Keep generated outputs tied to their inputs and preserve negative
results; do not update reference numbers just to make a check pass.

Never commit API keys, local environments, or restricted source data. Changes
to research claims need reproducible evidence, including hardware and snapshot
versions when relevant. CI checks correctness within its stated scope; it does
not establish performance on untested hardware or validate a new live dataset.

For a release, first merge a reviewed PR with passing CI, then tag the reviewed
commit and include its environment, input hashes, and remaining limitations in
the release notes. Do not tag unreviewed analysis as a verified research result.
