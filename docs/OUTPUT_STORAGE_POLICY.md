# Output and source storage

Maintained source/config/tests belong in their code owners. Current guides live
in docs, scientific synthesis in research, stable contracts in OpenSpec specs.
No historical source-object corpus or operation diary tree is maintained in HEAD.

Model checkpoints, datasets, generated outputs and run receipts remain in their
explicit external/local artifact roots. Do not traverse a symlink to remove them,
copy them into docs, or delete them as part of a tracked-file cleanup. Ignored
`.local` validation evidence is not a scientific dataset or an archive interface.

Decision-bearing current runs bind clean Git source plus independently specified
input/model/config/runtime facts. Old receipts lacking verifiable current source
cannot continue. Do not regenerate dirty-source snapshots, rewrite old hashes or
substitute current files for missing historical source. Recover details through
the catalog's exact commit/path when needed, then qualify a new run explicitly.

Publication must preserve its selected exclusive/idempotent contract. Changing a
codec, newline, token layout or coordinate scale is not a harmless storage rename.
Original artifact bytes and scientific labels remain historical truth; a technical
invalidity cannot be turned into a scientific null by cleaning the directory.
