# Structure of codebase

This codebase is to unify and standardize chemistry tools that could be helpful for molecular property prediction e.g. RDKit.
(2) "/utils" provides endpoints to individual features on a low-level that are then re-used for high-level full tools or intermediate endpoints that build upon individual features. These low-level endpoints should always have a cache for itself.
(2) "/cache" stores intermediate artfifacts that makes tools run faster.
    - We should re-use caches as much as possible.
(3) "/tools" provides versioning of full, properly-scoped tools that we provide to LLMs
    - It should use endpoints from "/utils" and perhaps have its own proper cache in case collecting features across multiple caches takes too much time
    - This should be versioned whenever we add new tools
    - Everytime we modify a tool or add a tool, we should run it on example molecules and output it to "/tool_auditing"

# Libraries and Models
molgpka (in the conda env openrlhf)
minimol (in the conda env minimol)

# API design philosophy

Use public APIs internally when the caller wants the same semantic operation that
external callers get. Do not create private duplicates of public APIs with nearly
the same behavior.

For the DuckDB-first feature path, `get_primitives(smiles, primitive_names,
params=None)` is the canonical primitive-value API. Internal bundle/full-feature
code should call `get_primitives` when it needs typed primitive values.
`get_primitive` may exist only as a tiny convenience wrapper around
`get_primitives(...)[name]`; do not build new internal flows around it.

Only v17 tool surfaces are current public LLM-facing APIs. Older versioned tool
surfaces belong under explicit legacy namespaces (`therapeutic_tools.tools.legacy`
or `therapeutic_tools.legacy.tools`) and should not be used by default public
routing.

Private helpers are appropriate only for narrower mechanics that are not the
public semantic operation: DuckDB row lookup/write, subprocess runtime
execution, rendering already-loaded primitive values, parsing, validation, and
test scaffolding. Name those helpers by the concrete mechanical job they do
(`lookup_primitives`, `write_primitive_rows`, `compute_missing_primitives`,
`render_bundle`) rather than vague terms like `ensure_primitives`.
