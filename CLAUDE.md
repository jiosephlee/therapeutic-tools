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