"""
Tool 6: Toxicophore Screening — semantic structural alert classification.

Groups raw RDKit filter catalog matches into mechanistically meaningful
toxicophore categories, each with a one-line note explaining why it matters.

Also screens against ToxAlerts (OCHEM) endpoint-specific SMARTS patterns
to provide direct task-relevant toxicity predictions.
"""

import os
import csv
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache


# ── Semantic toxicophore categories ─────────────────────────────────────
#
# Each category: (display_name, mechanistic_note, list_of_patterns)
# Patterns are matched case-insensitively against raw alert descriptions.
# Order matters — first match wins (an alert is assigned to one category).
#
# Pattern matching: each pattern is checked as a substring of the
# normalized (lowercased, underscores→spaces) alert description.

_TOXICOPHORE_CATEGORIES: List[Tuple[str, str, List[str]]] = [
    # ── Electrophilic reactivity ────────────────────────────────────
    (
        "Michael acceptor / electrophilic alkene",
        "Electrophilic alkene conjugated with electron-withdrawing group; "
        "can covalently modify cysteine residues on proteins.",
        ["michael", "alpha beta-unsaturated", "acrylate", "acrylonitrile",
         "vinyl_sulphone", "vinyl sulphone", "maleimide", "ene_one",
         "ene_quin_methide", "vinyl michael", "alkynyl michael",
         "trisub_bis_act_olefin", "vinyl_carbonyl_ewg",
         "activated_vinyl_ester", "activated_vinyl_sulfonate",
         "bis_keto_olefin", "ene_sulfone", "filter88_ene_sulfone",
         "propenal"],
    ),
    (
        "Alkylating agent",
        "Can transfer alkyl groups to DNA bases, causing mutations.",
        ["mustard", "nitrogen_mustard", "alkyl_halide", "alkyl halide",
         "r1 reactive alkyl halide", "allyl_halide", "benzyl_halide",
         "halo_olefin", "halo_acrylate", "beta halo carbonyl",
         "alpha_halo_carbonyl", "alpha halo carbonyl", "alpha_halo_ewg",
         "primary_halide_sulfate", "secondary_halide_sulfate",
         "tertiary_halide_sulfate", "halogenated_ring", "halogenated ring",
         "filter26_alkyl_halide", "filter30_beta_halo_carbonyl",
         "filter75_alkyl_br_i", "filter3_allyl_halide",
         "filter45_allyl_halide", "filter4_alpha_halo_carbonyl",
         "n-c-hal", "halo_imino", "alpha_halo_amine",
         "alpha_halo_heteroatom", "monofluoroacetate",
         "filter84_nitrogen_mustard",
         "beta-fluoro-ethyl-on"],
    ),
    (
        "Epoxide / aziridine / strained ring",
        "Strained 3-membered ring electrophile; reacts with DNA and proteins.",
        ["epoxide", "aziridine", "thioepoxide", "three-membered_heterocycle",
         "three-membered heterocycle", "three_membered_heterocycle",
         "filter40_epoxide_aziridine", "i6 epoxide",
         "activated_4mem_ring", "beta_lactone", "four_member_lactone",
         "four member lactone", "i14 four membered lactone",
         "propiolactone", "betalactam", "b-lactam", "i16 betalactam"],
    ),
    (
        "Aldehyde / reactive carbonyl",
        "Reactive carbonyl that forms Schiff bases with lysine residues; "
        "includes alpha-dicarbonyls and reactive ketone variants.",
        ["aldehyde", "filter38_aldehyde", "azoalkanal",
         "alpha_dicarbonyl", "1,2-dicarbonyl", "diketo_group", "diketo group",
         "filter41_12_dicarbonyl", "filter42_12_dicarbonyl",
         "filter72_hydrated_di_ketone", "ccl3-cho_releasing",
         "reactive_carbonyl", "oxalyl",
         "trichloromethyl_ketone", "trifluoromethyl_ketone",
         "perhalo_ketone", "keto_def_heterocycle",
         "meldrums_acid_deriv"],
    ),
    (
        "Nitroaromatic / nitro group",
        "Can be reduced to reactive nitroso/hydroxylamine intermediates "
        "that damage DNA.",
        ["nitro", "nitroso", "nitrosamine", "n-nitroso", "filter11_nitrosamin",
         "filter12_nitroso", "oxygen-nitrogen single bond",
         "oxygen-nitrogen_single_bond", "trinitro", "dinitrobenzene",
         "nitro aromatic", "aromatic no2", "filter77_alkyl_no2",
         "nitrosone_not_nitro", "nitrate"],
    ),
    (
        "Acyl halide / acid anhydride / activated ester",
        "Highly reactive acylating agents; react with nucleophilic "
        "residues non-selectively.",
        ["acid_halide", "acid halide", "acyl_halide", "acyl halide",
         "anhydride", "acid_anhydride", "filter2_acyl", "filter27_anhydride",
         "sulfonyl_halide", "sulfonyl halide", "sufonyl halide",
         "filter25_sulfonyl_halide", "carbonyl_halide",
         "hobt_ester", "hobt ester", "ester of hobt", "r10 esters of hobt",
         "pentafluorophenyl_ester", "pentafluorophenyl ester",
         "r8 pentafluorophenyl", "pentahalophenyl",
         "bis_activated_aryl_ester", "tris_activated_aryl_ester",
         "acyl_123_triazole", "acyl_134_triazole",
         "acyl_imidazole", "acyl_pyrazole", "acyl_activated_no",
         "activated_acetylene",
         "trifluoroacetate_ester", "trifluoroacetate_thioester",
         "trifluroacetate_amide"],
    ),
    # ── Genotoxic / mutagenic motifs ────────────────────────────────
    (
        "Polycyclic aromatic hydrocarbon (PAH)",
        "Planar polyaromatic system; metabolized by CYP1A1/1B1 to "
        "diol-epoxides that bind DNA.",
        ["polycyclic_aromatic", "polycyclic aromatic", "polynuclear_aromatic",
         "linear_polycyclic_aromatic", "pyrene", "phenanthrene", "phenalene",
         "filter63_polyaromatic", "filter68_anthracene", "branched_polycyclic",
         "multiple aromatic rings", "pah",
         "poly(azo(anthracene)", "poly(azo(phenanthrene)"],
    ),
    (
        "Azo compound",
        "Azo bonds can be reductively cleaved to release aromatic amines "
        "(potential carcinogens).",
        ["azo_a(", "azo_aryl", "azo_amino", "azo_filter", "azo group",
         "azo_group", "azobenzene", "azocyanamide", "filter5_azo",
         "p-aminoaryl_diazo", "dye "],
    ),
    (
        "Aniline / aromatic amine",
        "Aromatic amines can be metabolically activated (N-hydroxylation) "
        "to reactive species that bind DNA.",
        ["aniline", "analine", "anil_", "anil ",
         "benzidine", "diaminobenzene",
         "naphth_amino"],
    ),
    # ── Redox-active / ROS-generating ───────────────────────────────
    (
        "Quinone / hydroquinone",
        "Redox-active; generates reactive oxygen species and can "
        "deplete glutathione.",
        ["quinone", "chinone", "hydroquinone", "filter23_ortho_quinone",
         "filter53_para_quinone", "disulfonylimino", "ortho_hydroimino",
         "para_hydroimino"],
    ),
    (
        "Polyphenol / catechol",
        "Redox-active; auto-oxidizes to quinones and generates ROS.",
        ["catechol", "polyphenol", "dihydroxybenzene", "trihydroxyphenyl",
         "filter57_polyphenol", "filter58_polyphenol", "hydroquin_a",
         "polyhalo_phenol"],
    ),
    # ── Reactive sulfur / nitrogen ──────────────────────────────────
    (
        "Thiol / disulfide / thioether",
        "Reactive sulfur; can disrupt disulfide bonds in proteins "
        "or generate ROS.",
        ["thiol", "disulfide", "disulphide", "polysulfide", "filter56_ss_bond",
         "thioles_(not_aromatic)", "filter74_thiol",
         "diamino_sulfide", "conjugated_dithioether",
         "dithiomethylene_acetal", "thio_xanthate",
         "filter36_ss_double_bond"],
    ),
    (
        "Hydrazine / hydrazone / azide",
        "Can be metabolically activated to reactive radicals; genotoxic.",
        ["hydrazine", "hydrazone", "hydrazide", "acylhydrazide", "azide",
         "azido", "diazo", "diazonium", "filter20_hydrazine", "filter7_diazo",
         "carbazide", "any carbazide", "hzone_", "hzide_",
         "hydrazothiourea"],
    ),
    (
        "Isocyanate / isothiocyanate",
        "Electrophilic; reacts with nucleophilic amino acids "
        "(cysteine, lysine).",
        ["isocyanate", "isothiocyanate", "carbodiimide", "isonitrile",
         "filter8_thio_isocyanat", "iso(thio)cyanate"],
    ),
    # ── Sulfur-containing PAINS / liabilities ───────────────────────
    (
        "Thiourea / thioamide",
        "Thiocarbonyl compounds; can be metabolized to reactive sulfoxides "
        "or interfere with protein targets non-specifically.",
        ["thio_urea", "thiourea", "thioamide", "thio_amide",
         "thio_carbam", "filter81_thiocarbamate",
         "thio_imine_ium"],
    ),
    (
        "Rhodanine / thiazolidinone",
        "Known PAINS scaffold; promiscuous binder that gives false "
        "positives in many assays.",
        ["rhod_", "rhodanine", "thiazolidinone",
         "ene_rhod_", "rhod_sat_"],
    ),
    (
        "Thiocarbonyl / thioester / thiocyanate",
        "Reactive sulfur-carbon bonds; can be metabolized to toxic "
        "intermediates or react with biological nucleophiles.",
        ["thiocarbonyl", "thio_carbonate", "thiocarbonate",
         "thioester", "thio_ester", "thioesters", "filter29_thioester",
         "thioketone", "thio_ketone", "thio_keto_het",
         "thio_aldehyd", "thio_est_cyano",
         "thionoester", "thionyl",
         "thiocyanate", "thio cyanate", "thio_cyano",
         "i10 thiocyanate", "i12 thioester",
         "filter69_thio_carbonate", "filter73_thio_ketone",
         "filter76_s_ester",
         "aryl_thiocarbonyl",
         "s_or_o_c_triplebond_n", "filter67_s_or_o"],
    ),
    # ── Electrophilic heterocycles (PAINS) ──────────────────────────
    (
        "Imine / Schiff base",
        "C=N bond susceptible to hydrolysis; imines conjugated with "
        "carbonyls are frequent hitters in screening assays.",
        ["imine_one", "imine_ene", "imine_imine", "imine_naphthol",
         "imine_phenol", "imine_2", "imine_1", "imine 1", "imine 3",
         "imines", "acyclic imines", "acyclic_imide", "acyclic c=n-h",
         "filter39_imine", "filter24_react_imide", "filter78_bicyclic_imide",
         "imines_(not_ring)", "imine2"],
    ),
    (
        "Enamine / alkene-heterocycle conjugate",
        "Conjugated systems with heterocyclic rings; common PAINS motif "
        "that interferes with multiple assay types.",
        ["enamine", "alkyl enamine", "enamine_like",
         "ene_five_het", "ene_six_het", "ene_five_one",
         "ene_misc", "ene_cyano"],
    ),
    (
        "Mannich base",
        "Aminoalkyl carbonyl; can decompose to release formaldehyde "
        "and reactive iminium ions.",
        ["mannich"],
    ),
    # ── Cyanide / nitrile ───────────────────────────────────────────
    (
        "Acyl cyanide / reactive nitrile",
        "Highly electrophilic carbon attacked by biological nucleophiles; "
        "includes conjugated nitrile systems.",
        ["acyl_cyanide", "acyl cyanide", "sulfonyl_cyanide", "sulfonyl cyanide",
         "sulphonyl_cyanide",
         "cyanophosphonate", "cyanohydrin", "filter21_cyanhydrin",
         "cyanamide", "cyano_phosphonate",
         "conjugated_nitrile", "conjugated nitrile",
         "aminonitrile", "geminal_dinitriles", "four_nitriles",
         "cyanate", "filter70_alkylcn2",
         "cyano_amino_het", "cyano_cyano", "cyano_ene_amine",
         "cyano_imine", "cyano_keto", "cyano_misc",
         "cyano_pyridone"],
    ),
    # ── Metal / organometallic ──────────────────────────────────────
    (
        "Heavy metal / organometallic / boron",
        "Metals and metalloids can inhibit enzymes by binding to "
        "active-site residues.",
        ["heavy_metal", "heavy metal", "contains_metal", "filter9_metal",
         "metal_carbon_bond", "unacceptable atoms",
         "bad_boron", "boron_warhead", "bad_cations",
         "si,b,se", "hetero_silyl", "silicon_halogen", "silicon halogen",
         "triphenyl_silyl", "triphenyl_methyl-silyl", "triphenyl methylsilyl",
         "triphenyl_boranyl",
         "filter37_silicate",
         "undesirable_elements"],
    ),
    # ── Oxidizing / unstable ────────────────────────────────────────
    (
        "Peroxide / hydroxamic acid / N-oxide",
        "Oxidizing agents; can generate free radicals or be "
        "metabolically activated.",
        ["peroxide", "oxime", "hydroxamic_acid", "hydroxamic acid",
         "hydroxamate", "filter32_oo_bond", "filter18_oxime_ester",
         "triacyloxime",
         "n_oxide", "n oxide", "n-oxide",
         "aromatic_n-oxide", "filter89_hydroxylamine",
         "aminooxy(oxo)"],
    ),
    # ── Phosphorus ──────────────────────────────────────────────────
    (
        "Phosphorus compound",
        "Reactive phosphorus groups; phosphoramides and phosphonates can "
        "alkylate DNA or inhibit cholinesterases.",
        ["phosphor", "phosphoramide", "phosphonate", "phosphonium",
         "phosphorane", "phosphene", "phosphite",
         "no_phosphonate", "active_phosphate",
         "di_and_triphosphate", "di/triphosphate", "diand_triphosphate",
         "i15 di and triphosphate", "tri phosphoric",
         "p/s halide", "p/s_halide", "p_or_s_halide",
         "r14 phosphoramide", "r19 phosphorane", "r22 p/s halide",
         "filter16_trialkyl_phosphin", "filter17_trialkyl_phosphin",
         "filter13_ps_double_bond", "filter35_pp_bond",
         "filter61_phosphor_halide", "filter48_foso", "filter51_pn3",
         "phosphorus_halide", "phosphorus_phosphorus",
         "phosphorus_sulfur", "phos_serine_warhead",
         "phos_threonine_warhead", "phos_tyrosine_warhead",
         "ugly p compound", "thiophosphothionate"],
    ),
    # ── Halogenated compounds ───────────────────────────────────────
    (
        "Polyhalogenated / perfluorinated",
        "Heavily halogenated compounds; may accumulate in tissue, "
        "disrupt membranes, or release toxic metabolites.",
        ["perfluorinated", "perfluoralkyl", "per_halo_chain",
         "filter66_c4_perfluoralkyl", "filter83_per_halo_chain",
         "high halogen", "gte_7_total_hal",
         "fluorinated_carbon", "fluorines",
         "polyhalo_phenol", "perchloro",
         "perchlorates", "chlorates",
         "halo_5heterocycle_bis_ewg",
         "halo_phenolic_carbonyl", "halo_phenolic_sulfonyl",
         "filter64_halo_ketone_sulfone",
         "halogen_heteroatom", "filter46_nhalide", "n-halo",
         "filter52_nc_haloamine", "n-haloamine",
         "trihalovinyl_heteroatom",
         "aryl bromide", "aryl iodide", "iodine", "gte_3_iodine",
         "cl,br,i"],
    ),
    # ── Crown ethers ────────────────────────────────────────────────
    (
        "Crown ether",
        "Macrocyclic polyethers that sequester metal cations; can disrupt "
        "ion homeostasis and interfere with biological processes.",
        ["crown_ether", "crown ether", "crown_ethers",
         "i3 crown", "filter87_crown",
         "poly ether"],
    ),
    # ── Sulfonyl / sulfonate / sulfate ──────────────────────────────
    (
        "Sulfonate / sulfate ester",
        "Sulfonates and sulfate esters can act as alkylating agents; "
        "sulfonamides are common pharmacophores but some interfere "
        "with assays.",
        ["sulfonate", "sulphonate", "sulfonates", "sulphonates",
         "sulfate_ester", "sulphate_ester", "sulphate", "sulfate ester",
         "r4 sulphate ester", "r5 sulphonate",
         "sulfonic_acid", "sulfonic acid", "sulfinic_acid", "sulfinic acid",
         "heteroaryl sulfonate", "aromatic sulfonic ester",
         "filter65_alkyl_sulfonate",
         "triflate", "triflates", "r12 triflate",
         "aliphatic_triflate",
         "sulfite_sulfate_ester",
         "sulf_d2_oxygen_d2",
         "sulfonamide_a", "sulfonamide_b", "sulfonamide_c",
         "sulfonamide_d", "sulfonamide_e", "sulfonamide_f",
         "sulfonamide_g", "sulfonamide_h", "sulfonamide_i",
         "sulfonamide_j",
         "dicarbonyl_sulfonamide",
         "sulfonium", "filter22_sulfonium",
         "sulfonyl_heteroatom",
         "sulfur_oxygen_single_bond", "sulfur oxygen single bond",
         "filter31_so_bond", "filter47_so2f",
         "hyperval_sulfur", "thiosulfoxide",
         "filter14_thio_oxopyrylium", "filter15_thiosulfate",
         "s/po3 group"],
    ),
    # ── Ester / acetal / carbonate ──────────────────────────────────
    (
        "Ester / phenol ester / carbonate",
        "Esters can be hydrolyzed to release active metabolites; "
        "phenol esters and carbonates may indicate prodrug instability.",
        ["phenol_ester", "phenol ester", "phenylester",
         "phenyl_carbonate", "phenyl carbonate",
         "> 2 ester group", ">_2_ester_group",
         "ester", # catches "Ester" and "aliphatic ester"
         "acetal", "non_ring_acetal", "non_ring_ketal",
         "non_ring_ch2o_acetal", "hemiacetal",
         "orthoester",
         "filter19_hydroxyimide_ester",
         "filter93_acetyl_urea"],
    ),
    # ── Heterocyclic PAINS ──────────────────────────────────────────
    (
        "Thiazole / thiophene (PAINS)",
        "Sulfur-containing heterocycles flagged as frequent hitters "
        "in high-throughput screening.",
        ["thiaz_ene", "thiazol_sc",
         "thiazole_amine",
         "thiophene_amino", "thiophene_hydroxy",
         "thiophene_c(", "thiophene_d(", "thiophene_e(", "thiophene_f(",
         "het_thio_5", "het_thio_6", "het_thio_n",
         "het_thio_pyr", "het_thio_urea_ene",
         "thio_dibenzo", "thio_thiomorph",
         "thiobenzothiazole"],
    ),
    (
        "Pyrrole / indole (PAINS)",
        "Electron-rich heterocycles prone to oxidation; some variants "
        "are frequent hitters in screening assays.",
        ["pyrrole_", "indol_3yl", "indole_3yl",
         "misc_pyrrole"],
    ),
    (
        "Coumarin / flavonoid",
        "Polyphenolic heterocycles; known assay interferers that can "
        "auto-fluoresce or aggregate.",
        ["coumarin", "cumarine", "flavanoid", "flavin", "fluorescein",
         "colchicine",
         "anthranil_acid", "anthranil_amide", "anthranil_one",
         "porphyrin"],
    ),
    (
        "Dihydropyridine (PAINS)",
        "DHP scaffolds are frequent hitters; some are redox-active "
        "and interfere with multiple assay formats.",
        ["dhp_amino", "dhp_amidine", "dhp_keto",
         "dhp_bis_amino"],
    ),
    (
        "Pyridinium / quaternary nitrogen",
        "Permanent positive charge can cause non-specific binding "
        "to proteins and membranes.",
        ["pyridinium", "het_pyridinium", "filter82_pyridinium",
         "quaternary_nitrogen", "quaternary nitrogen", "quaternary_n",
         "quaternary_c,cl,i,p,s", "quaternary",
         "r18 quaternary",
         "benzylic_quaternary", "beta_carbonyl_quaternary",
         "quat_n_n", "quat_n_acyl",
         "gte_2_n_quat",
         "paraquat", "oxonium", "imidazolium",
         "thiopyrylium", "pyrylium"],
    ),
    # ── Miscellaneous structural flags ──────────────────────────────
    (
        "Long aliphatic chain",
        "Molecules with long alkyl chains (>=7 carbons) tend to be "
        "non-specific membrane disruptors with poor drug-likeness.",
        ["aliphatic_long_chain", "aliphatic long chain",
         "long aliphatic chain", "long chain hydrocarbon",
         "unbranched chain", "heptane",
         "gte_10_carbon", "gte_8_cf2_or_ch2",
         "i1 aliphatic methylene", "filter33_c10_alkyl"],
    ),
    (
        "Styrene / vinyl / alkene",
        "Terminal or conjugated alkenes can be metabolically epoxidized "
        "to form reactive intermediates.",
        ["styrene", "stilbene", "phenylethene",
         "terminal vinyl", "filter10_terminal_vinyl",
         "isolated_alkene", "isolated alkene",
         "acyclic c=c-o", "acyclic_c=c-o",
         "ethene", "vinyl_halide", "diene", "enyne",
         "allene", "diacetylene", "polyene",
         "n2 polyene", "adjacent_ring_double_bond",
         "polyene_chain_between_aromatic", "polyines",
         "cyclobutene", "ring_triple_bond", "triple_bond", "triple bond",
         "activated_acetylene"],
    ),
    (
        "Acridine / intercalator",
        "Planar heterocycles that intercalate into DNA; can cause "
        "frameshift mutations.",
        ["acridine", "amino_acridine", "amino_naphtalimide",
         "azulene"],
    ),
    (
        "Natural product toxin scaffold",
        "Structural motifs from known natural toxins.",
        ["saponin", "saponine", "n3 saponin",
         "cytochalasin", "n4 cytochalasin",
         "cycloheximide", "n5 cycloheximide",
         "monensin", "n6 monensin",
         "cyanidin", "n7 cyanidin",
         "squalestatin", "n8 squalestatin",
         "colchicine",
         "biotin_analogue", "biotin analogue"],
    ),
    (
        "Miscellaneous heterocyclic PAINS",
        "Various heterocyclic scaffolds flagged as frequent hitters "
        "in high-throughput screening assays.",
        ["het_5_", "het_55_", "het_65_", "het_66_",
         "het_465", "het_565", "het_666", "het_6666",
         "het_76_",
         "het_6_hydropyridone", "het_6_imidate",
         "het_6_pyridone", "het_6_tetrazine",
         "het_pyraz_misc",
         "imidazole_a(", "imidazole_b(", "imidazole_c(",
         "imidazole_amino",
         "furan_a(", "furan_acid",
         "misc_furan", "misc_imidazole",
         "acyl_het_a(",
         "anisol_a(", "anisol_b(",
         "misc_anisole", "misc_anilide",
         "misc_naphthimidazole",
         "misc_phthal_thio", "misc_pyridine",
         "misc_pyrrole_thiaz", "misc_stilbene",
         "misc_trityl", "misc_urea", "misc_aminoacid",
         "misc_aminal", "misc_cyclopropane",
         "keto_keto_beta", "keto_keto_gamma",
         "keto_phenone", "keto_naphthol",
         "keto_thiophene",
         "phenol_sulfite",
         "steroid_a(",
         "phthalimide", "hydantoin",
         "tetrazole", "triazole", "amidotetrazole",
         "aminothiazole",
         "tetraazinane", "thiatetrazolidine",
         "oxobenzothiepine", "tropone",
         "oxepine", "pyranone", "azepane", "azocane",
         "cycloheptane", "cyclooctane",
         "melamine",
         "dyes3a(", "dyes5a(", "dyes7a(",
         "het-c-het", "het c-het"],
    ),
    # ── Catch-all physicochemical flags ──────────────────────────────
    (
        "Reactive / unstable group",
        "Miscellaneous reactive or unstable functional groups "
        "flagged by medicinal chemistry filters.",
        ["lawesson", "ketene", "formate_formide",
         "glycol", "carbamate",
         "triamide", "acyclic_imide",
         "geminal_amine", "noname",
         "oxy-amide", "charged_oxygen_or_sulfur",
         "charged oxygen or sulfur",
         "o-tertbutylphenol", "tert_butyl",
         "adamantyl", "benzhydrol",
         "double_trouble_warhead",
         "dipeptide", "amino acid", "amino_acid",
         "bis_amino",
         "gte_", "too many",
         "phenolate_bis_ewg", "poly_sub_atomatic",
         "pcp", "amino_acridine",
         "misc_",
         "iodine", "isotope", "filter34_isotope", "deuterium",
         "carbons", "non-hydrogen_atoms", "n,o,s",
         "chloramidine", "chloramidines", "r20 chloramidine",
         "filter62_oxo_thio_halide",
         "filter49_halogen",
         "filter50_grignard",
         "filter59_phoshorous_ylide",
         "filter60_acyclic_n-s", "acyclic n-s", "acyclic n-c-n",
         "acyclic n-,=n",
         "s=n_(not_ring)", "filter90_n_double_bond_s",
         "n:c-sch2",
         "halo_pyrazine", "halo_pyridazine", "halo_pyridine",
         "halo_pyrimidine", "2-halo", "4-halo", "2-chloro",
         "filter28_halo_pyrimidine", "filter94_2_halo_pyridine",
         "4_pyridone_3_5_ewg",
         "triaryl_phosphine_oxide", "triphenylphosphine",
         "n-hydroxyl_pyridine", "n-hydroxyl pyridine",
         "ch2_s#o_3_ring", "activated_s#o_3_ring",
         "no_phosphonate",
         "c13", "26", "28", "42", "43",
         "aliphatic ketone",
         "conjugated nitrile", "conjugated_nitrile",
         "cyanate_/aminonitrile", "cyanate/aminonitrile",
         "n:c-sch2 groups",
         "alkyl esters of s or p",
         "hetero imide",
         "2halo_"],
    ),
]

# Pre-compile patterns for speed
_COMPILED_CATEGORIES = [
    (name, note, [p.lower() for p in pats])
    for name, note, pats in _TOXICOPHORE_CATEGORIES
]


def _classify_alert(description: str) -> str | None:
    """Map a raw alert description to a semantic category name, or None."""
    desc_lower = description.strip().lower().replace("_", " ")
    for cat_name, _, patterns in _COMPILED_CATEGORIES:
        for pat in patterns:
            pat_norm = pat.replace("_", " ")
            # Pattern is substring of description (normal case)
            # OR description is exact match of a pattern
            if pat_norm in desc_lower or desc_lower == pat_norm:
                return cat_name
    # Exact-match table for short/ambiguous descriptions
    return _EXACT_MATCH_MAP.get(desc_lower)


# Exact matches for short descriptions that can't be caught by substring
_EXACT_MATCH_MAP = {
    "azo": "Azo compound",
    "imine": "Imine / Schiff base",
    "ketone": "Aldehyde / reactive carbonyl",
    "carbo cation/anion": "Reactive / unstable group",
    "carbocation/anion": "Reactive / unstable group",
    "hetero hetero": "Reactive / unstable group",
    "polyacidic": "Reactive / unstable group",
    "i2 compounds with 4 or more acidic groups": "Reactive / unstable group",
    "n-s (not sulfonamides)": "Reactive / unstable group",
    "sulphur halide": "Reactive / unstable group",
    "tri pentavalent s": "Reactive / unstable group",
    "thiomorpholinedione": "Thiocarbonyl / thioester / thiocyanate",
    "filter1 2 halo ether": "Alkylating agent",
    "filter92 trityl": "Reactive / unstable group",
    "perhalo dicarbonyl phenyl": "Polyhalogenated / perfluorinated",
    "perhalo phenyl": "Polyhalogenated / perfluorinated",
    "pyrazole amino a(1)": "Miscellaneous heterocyclic PAINS",
    "pyrazole amino b(1)": "Miscellaneous heterocyclic PAINS",
    "thio ene amine a(1)": "Thiazole / thiophene (PAINS)",
    "thio imide a(1)": "Thiocarbonyl / thioester / thiocyanate",
    "thio pyridine a(1)": "Thiazole / thiophene (PAINS)",
}


# ── ToxAlerts (OCHEM) endpoint-specific screening ─────────────────────

_TOXALERTS_ENDPOINT_MAP = {
    "Skin sensitization": (
        "Skin sensitization",
        "Structural alerts for skin sensitization via haptenation "
        "(covalent binding to skin proteins)."
    ),
    "Genotoxic carcinogenicity, mutagenicity": (
        "Genotoxic carcinogenicity / mutagenicity",
        "DNA-reactive motifs that cause mutations via direct genotoxic mechanisms."
    ),
    "Non-genotoxic carcinogenicity": (
        "Non-genotoxic carcinogenicity",
        "Promotes cancer through non-DNA-reactive mechanisms "
        "(receptor activation, epigenetic changes, chronic inflammation)."
    ),
    "Idiosyncratic toxicity (RM formation)": (
        "Reactive metabolite formation (idiosyncratic toxicity)",
        "Can form reactive metabolites via CYP metabolism that covalently "
        "modify liver proteins, triggering immune-mediated hepatotoxicity."
    ),
    "Reactive, unstable, toxic": (
        "Reactive / unstable / toxic groups",
        "Generally reactive or unstable functional groups "
        "flagged across multiple toxicity endpoints."
    ),
    "Potential electrophilic agents": (
        "Electrophilic agents",
        "Electrophilic centers that can react with biological nucleophiles "
        "(protein thiols, DNA bases)."
    ),
    "Chelating agents": (
        "Chelating agents",
        "Can sequester metal ions essential for enzyme function; "
        "may interfere with ion channels."
    ),
    "Developmental and mitochondrial toxicity": (
        "Developmental / mitochondrial toxicity",
        "Disrupts mitochondrial function or embryonic development."
    ),
    "Promiscuity": (
        "Promiscuous compounds",
        "Structural motifs associated with non-specific activity across "
        "many biological targets (frequent hitters)."
    ),
    "Acute Aquatic Toxicity": (
        "Acute aquatic toxicity",
        "Structural features associated with acute toxicity to aquatic "
        "organisms, indicating general ecotoxicity and bioactivity."
    ),
}

_TOXALERTS_SKIP = {
    "PAINS compounds", "Extended Functional Groups (EFG)", "Custom filters",
    "AlphaScreen-GST-FHs", "AlphaScreen-HIS-FHs", "AlphaScreen-FHs",
    "Biodegradable compounds", "Nonbiodegradable compounds",
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


import re as _re

def _fix_greek_letters(name: str) -> str:
    """Restore Greek letters corrupted to '?' in ToxAlerts CSV export."""
    # α,β-Unsaturated (two-letter pattern)
    name = name.replace("?, ?-", "α,β-").replace("?,?-", "α,β-").replace("?,? ", "α,β-")
    name = _re.sub(r"\(?,\?-unsaturated", "(α,β-unsaturated", name)
    # 1,2 ? / 1,3 ? / 1,4 ? → 1,2- / 1,3- / 1,4- (lost dash)
    name = _re.sub(r"(\d,\d) \?[ -]", r"\1-", name)
    # Context-specific single ? replacements
    _BETA = [
        "?-Lactam", "?-Lacton", "?-Diketone", "?-dithione",
        "?-Naphthol", "?-Propiolactone", "?-Aminonaphtalene",
        "?- sultone", "?-sultone",
    ]
    _ALPHA = [
        "?-Halo", "?-halo", "?-Carbonyl", "?-Amino", "?-amino",
        "?-Hydroxy", "?-hydroxy", "?-Oxo",  "?-Substituted",
        "?-posi", "?-halogen",
    ]
    for pat in _BETA:
        name = name.replace(pat, "β" + pat[1:])
    for pat in _ALPHA:
        name = name.replace(pat, "α" + pat[1:])
    # Remaining standalone ? likely Greek — remove stray ones
    name = name.replace("?yano", "Cyano")
    name = name.replace("valence?states", "valence states")
    name = name.replace("compounds?", "compounds")
    name = name.replace("DDT?", "DDT ")
    return name


@lru_cache(maxsize=1)
def _load_toxalerts():
    """Load and compile ToxAlerts SMARTS patterns from CSV."""
    from rdkit import Chem

    csv_path = os.path.join(_CACHE_DIR, "toxalerts.csv")
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r") as f:
        lines = [l for l in f if l.strip()]

    reader = csv.DictReader(lines)
    alerts = []
    for r in reader:
        smarts = (r.get("SMARTS") or "").strip()
        prop = (r.get("PROPERTY") or "").strip()
        # Fix corrupted Greek letters (α/β lost during CSV export)
        name_raw = (r.get("NAME") or "").strip()
        name_raw = _fix_greek_letters(name_raw)
        if not smarts or smarts == "[ERROR]" or prop in _TOXALERTS_SKIP:
            continue
        if prop not in _TOXALERTS_ENDPOINT_MAP:
            continue
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            alerts.append((
                (r.get("Alert ID") or "").strip(),
                name_raw,
                prop,
                pat,
            ))
    return alerts


def _screen_toxalerts_data(smiles: str) -> Dict[str, Dict]:
    """Screen against ToxAlerts and return structured data.

    Returns dict mapping display_name -> {"note": str, "alerts": list[str]}.
    """
    from rdkit import Chem
    from collections import OrderedDict

    alerts = _load_toxalerts()
    if not alerts:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    grouped: OrderedDict[str, list] = OrderedDict()
    for alert_id, name, prop, pat in alerts:
        if mol.HasSubstructMatch(pat):
            if prop not in grouped:
                grouped[prop] = []
            grouped[prop].append(name)

    result = {}
    for prop, alert_names in grouped.items():
        display_name, note = _TOXALERTS_ENDPOINT_MAP[prop]
        unique_names = list(dict.fromkeys(alert_names))
        result[display_name] = {"note": note, "alerts": unique_names}

    return result


# ── Main entry point ──────────────────────────────────────────────────

def screen_toxicophores(smiles: str) -> str:
    """
    Screen a molecule for toxicophores (structural alerts grouped by
    mechanism of toxicity) and pharmacophore features.

    Alerts from multiple filter libraries (PAINS, Brenk, NIH, ZINC, ChEMBL)
    are deduplicated and grouped into mechanistic categories such as
    'Michael acceptor', 'alkylating agent', or 'PAH', each with an
    explanation of why the pattern is toxicologically relevant.

    Additionally screens against ToxAlerts (OCHEM) endpoint-specific
    SMARTS patterns to identify alerts linked to specific toxicity
    endpoints (skin sensitization, mutagenicity, DILI, etc.).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with toxicophore analysis.
    """
    sections = []

    # Collect structured data from both sources
    rdkit_data = {}
    toxalerts_data = {}
    try:
        rdkit_data = _screen_structural_alerts_data(smiles)
    except Exception as e:
        sections.append(f"Structural Alerts: Error screening RDKit catalogs - {e}")
    try:
        toxalerts_data = _screen_toxalerts_data(smiles)
    except Exception as e:
        sections.append(f"Structural Alerts: Error screening ToxAlerts - {e}")

    # Merge into unified "Structural Alerts" section
    # RDKit categories first, then ToxAlerts endpoint categories
    merged = dict(rdkit_data)
    for cat_name, data in toxalerts_data.items():
        if cat_name in merged:
            # Merge alerts, keeping existing note
            existing_alerts = set(a.lower() for a in merged[cat_name]["alerts"])
            for a in data["alerts"]:
                if a.lower() not in existing_alerts:
                    merged[cat_name]["alerts"].append(a)
                    existing_alerts.add(a.lower())
        else:
            merged[cat_name] = data

    if not merged:
        sections.append("Structural Alerts:\nNo structural alerts found.")
    else:
        lines = [f"Structural Alerts ({len(merged)} categories):"]
        for i, (cat_name, data) in enumerate(merged.items(), 1):
            alert_names = data["alerts"]
            lines.append("")
            lines.append(f"{i}. {cat_name}")
            lines.append(data["note"])
            shown = alert_names[:8]
            trail = f", ... and {len(alert_names) - 8} more" if len(alert_names) > 8 else ""
            lines.append(f"Matched alerts ({len(alert_names)}): {', '.join(shown)}{trail}")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def _screen_structural_alerts_data(smiles: str) -> Dict[str, Dict]:
    """Screen against all RDKit alert catalogs and return structured data.

    Returns dict mapping category_name -> {"note": str, "alerts": list[str]}.
    """
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from collections import OrderedDict

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
    fc = FilterCatalog(params)

    # Collect all raw alerts (deduplicated)
    seen = set()
    raw_alerts = []
    for entry in fc.GetMatches(mol):
        desc = entry.GetDescription().strip()
        desc_norm = desc.lower().replace("_", " ")
        if desc_norm not in seen:
            seen.add(desc_norm)
            raw_alerts.append(desc)

    if not raw_alerts:
        return {}

    # Group into semantic categories
    grouped: OrderedDict[str, Dict] = OrderedDict()
    uncategorized = []

    for alert in raw_alerts:
        cat = _classify_alert(alert)
        if cat:
            if cat not in grouped:
                note = ""
                for name, n, _ in _TOXICOPHORE_CATEGORIES:
                    if name == cat:
                        note = n
                        break
                grouped[cat] = {"note": note, "alerts": []}
            grouped[cat]["alerts"].append(alert)
        else:
            uncategorized.append(alert)

    if uncategorized:
        grouped["Other structural flags"] = {
            "note": "Miscellaneous structural alerts from medicinal chemistry filters.",
            "alerts": uncategorized,
        }

    return grouped


# Keep backward compatibility
screen_safety = screen_toxicophores


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "screen_toxicophores",
        "description": (
            "Screen a molecule for toxicophores — structural motifs associated with "
            "specific toxicity mechanisms. Groups alerts into categories like 'Michael "
            "acceptor' (protein-reactive electrophile), 'alkylating agent' (DNA-reactive), "
            "'PAH' (CYP-activated), etc. Each category includes a mechanistic explanation. "
            "Also screens against ToxAlerts endpoint-specific patterns covering skin "
            "sensitization, mutagenicity, carcinogenicity, reactive metabolite formation, "
            "electrophilic reactivity, chelation, and promiscuity. Returns pharmacophore "
            "feature counts (donor/acceptor/aromatic/hydrophobe). Use this to identify "
            "structural liabilities and understand their toxicological mechanisms."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."}
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }
    }
}
