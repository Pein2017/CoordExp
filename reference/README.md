# Retained execution evidence

`retained-sources/` holds exact bytes required by existing source-bound receipts.
It is not a code library, research reading path, or default search/index input.
Current implementation belongs in `src/` and `probes/`; useful conclusions belong
with their current research owner. Temporary code without a current consumer or
specific evidence obligation should be deleted, not archived here.

`objects/<sha-prefix>/<sha><suffix>` stores source bytes once. `runs/` keeps
receipt-bound names as hard links to those objects, preserving old paths and
checksums without copying identical bytes for every run. These generated aliases
are local-only. Captures are read-only evidence; never edit or execute them.
Historical relocation identities remain in `manifests/documentation-layout.json`.

The unused `legacy_src/` and `legacy_openspec_2026-06-29/` trees were removed.
Their exact tracked versions remain recoverable from Git commit
`f47c20606864f453e4ab8bb6a143319efa043b0c`; they have no maintained runtime consumer.
Do not restore the trees merely to satisfy an old documentation pointer.

Ordinary search and Codegraph/Cursor indexing exclude `reference/`. For an explicit
historical investigation, use a known receipt/path/hash and opt in to that path
(e.g. `rg --no-ignore pattern reference/retained-sources/objects/`).
