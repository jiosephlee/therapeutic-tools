# Potentially Building Our Own SoM Tool

Ideas and resources for expanding ATTNSOM's training data beyond the Zaretzki dataset (679 molecules, 9 CYP isoforms),
and a near-term alternative using GLORYx.

## Current ATTNSOM Weaknesses

- Small dataset: 679 unique molecules across 9 isoforms (~2003 molecule-isoform graphs total)
- Rare molecules (e.g., caffeine with 4/2003 graphs) produce weak gradient signal
- All from a single source (Zaretzki/XenoSite), limiting chemical diversity
- Even if ATTNSOM SoM predictions were perfect, integrating them with SyGMa metabolite generation
  to rank metabolites by feasibility (i.e., which metabolites actually form based on where CYPs attack)
  is a non-trivial engineering effort that hasn't been done yet
- FAME3R (trained on MetaQSAR, ~2000 substrates) is an alternative SoM model available for academic
  licensing from University of Milan — could be used instead of expanding ATTNSOM, but has same
  integration gap with metabolite prediction

## Near-Term: GLORYx API (Recommended First Step)

GLORYx already solves the SoM→metabolite integration problem: it uses FAME3 for SoM prediction
and applies reaction rules to generate ranked metabolite structures with feasibility scores.
Rather than building our own integrated pipeline, we should cache GLORYx predictions first.

### API Details (NERDD Platform)

The NERDD REST API follows a uniform pattern across all modules:

```python
import time, requests

# 1. Submit job
response = requests.post(
    "https://nerdd.univie.ac.at/api/gloryx/jobs",
    data={"inputs": ["CCO", "c1ccccc1"]},
).json()
job_id = response["id"]

# 2. Poll for completion
while True:
    status = requests.get(
        f"https://nerdd.univie.ac.at/api/jobs/{job_id}"
    ).json()
    if status["status"] == "completed":
        break
    time.sleep(5)

# 3. Fetch results (paginated, page=1 based)
results = requests.get(
    f"https://nerdd.univie.ac.at/api/jobs/{job_id}/results?page=1"
).json()
for result in results["data"]:
    print(result)

# 4. Clean up
requests.delete(f"https://nerdd.univie.ac.at/api/jobs/{job_id}")
```

- **Batch support**: POST with `data={"inputs": [list_of_SMILES]}` or file upload
- **Pagination**: Results served in pages (`page=1` to `num_pages_total`)
- **Incomplete results**: `?incomplete=true` returns partial results while job runs
- **No auth required**: Free for academic and commercial use
- **Rate limit**: Low bandwidth (public academic server) — need to be respectful, batch carefully

### Also available on NERDD: CYPstrate, FAME3R

- **CYPstrate** (`/api/cypstrate/jobs`): Substrate/non-substrate classification for 9 CYP isoforms
  (RF + SVM, trained on 1831 compounds). Params: `prediction_mode="best_performance"`
- **FAME3R** (`/api/fame3r/jobs`): Phase 1 + Phase 2 SoM prediction (atom-level probabilities)

### Target Tasks for GLORYx Cache

Of our 16 TDC tasks, only those where metabolism reasoning is relevant should be cached.

**Include (8 tasks):**
- `CYP2C9_Substrate_CarbonMangels` — direct CYP task, SoM + metabolites critical
- `CYP2D6_Substrate_CarbonMangels` — direct CYP task
- `CYP3A4_Substrate_CarbonMangels` — direct CYP task
- `DILI` — reactive metabolites cause hepatotoxicity
- `AMES` — metabolic activation of pro-mutagens
- `Carcinogens_Lagunin` — metabolic activation to carcinogenic species
- `ClinTox` — metabolic toxicity pathway
- `Bioavailability_Ma` — first-pass metabolism affects oral bioavailability

**Exclude (8 tasks):**
- `BBB_Martins` — permeability/efflux, not metabolism-driven
- `PAMPA_NCATS` — passive diffusion assay, purely physical
- `Pgp_Broccatelli` — transporter substrate, not metabolism
- `hERG` — ion channel binding, structure-based
- `HIA_Hou` — intestinal absorption, primarily permeability
- `SARSCoV2_3CLPro_Diamond` — target binding assay
- `SARSCoV2_Vitro_Touret` — in vitro antiviral activity
- `Skin_Reaction` — borderline (haptenation could involve metabolites, but weak signal)

### Cache Building Strategy

1. Collect unique SMILES from train+val+test of the 8 target tasks
2. Deduplicate across tasks (many molecules appear in multiple datasets)
3. Submit in batches to GLORYx API (respect rate limits, ~5s poll interval)
4. Store results as JSONL cache keyed by canonical SMILES
5. At inference time, look up cached GLORYx result and format for the model

## Long-Term: Expanding ATTNSOM Training Data

If GLORYx caching proves too slow or we need real-time inference, expanding ATTNSOM and
building our own SoM→metabolite integration pipeline becomes necessary.

### Candidate Datasets

### 1. AZ-ExactSOM (AstraZeneca)
- **Size**: 120 compounds with exact atom-level SoM annotations
- **Source**: Zenodo (DOI: 10.5281/zenodo.15458630)
- **License**: Open access
- **Pros**: High-quality pharma-grade annotations, immediately usable (SDF with atom-level labels)
- **Cons**: Small, may overlap with Zaretzki

### 2. MetaQSAR Database
- **Size**: ~2,000 substrates with SoM annotations across multiple CYP isoforms
- **Source**: University of Milan (Prof. Vistoli's group)
- **License**: Academic license required — must request access
- **Pros**: Large, high-quality, used to train FAME 1/2/3 models
- **Cons**: Not freely downloadable, requires academic agreement
- **Reference**: Pedretti et al., "MetaQSAR: An Integrated Database Engine to Manage and Analyze Metabolic Data"

### 3. AutoSOM Pipeline (Automated SoM Extraction)
- **Source**: GitHub — molinfo-vienna/AutoSOM
- **What it does**: Automatically extracts atom-level SoM annotations from substrate→metabolite SMILES pairs using MCS (Maximum Common Substructure) alignment
- **Pros**: Can generate SoM labels from any metabolite database (e.g., DrugBank metabolite pairs, HMDB)
- **Cons**: Inferred labels (not experimentally validated), quality depends on MCS alignment accuracy
- **Potential**: Could dramatically expand training data by mining public metabolite databases

### 4. Curated CYP450 Interaction Dataset
- **Source**: Nature Scientific Data
- **Size**: ~2,000 compounds per isoform
- **Pros**: Peer-reviewed, large, isoform-specific
- **Cons**: May focus on inhibition/substrate classification rather than atom-level SoM; need to verify annotation granularity

## Integration Strategy

1. **Quick win**: Add AZ-ExactSOM (120 compounds, open access, immediate)
2. **Medium-term**: Request MetaQSAR access, integrate with deduplication against Zaretzki
3. **Scalable approach**: Use AutoSOM to mine DrugBank/HMDB substrate→metabolite pairs for additional training data
4. **Validation**: Hold out a fraction of each new dataset for independent testing to verify generalization gains

## Key Considerations

- **TDC exclusion**: Any new training molecules must be checked against `tdc_exclusion_smiles.json` (122,731 SMILES)
- **Annotation format**: ATTNSOM expects SDF files with `PRIMARY_SOM` property (1-based atom indices). New datasets may need conversion.
- **Isoform coverage**: Zaretzki covers 9 isoforms. New datasets should ideally cover the same set, or we handle missing isoforms gracefully.
- **Chemical diversity**: Priority should be on molecules structurally dissimilar to current training set (low Tanimoto to existing), to improve applicability domain coverage.
