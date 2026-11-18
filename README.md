# TDP-43 Molecular Screening with XGBoost

Exploratory Python workflows for preparing molecular features from ChEMBL, training an XGBoost activity classifier, and screening SMILES strings for candidate compounds. The repository also contains early scripts for ADMET filtering and TDP-43 related analyses.

> **Project status:** Research prototype. The scripts are exploratory and should not be treated as a validated drug-discovery pipeline or as evidence of therapeutic activity.

## What’s here

| File | Purpose |
|---|---|
| `xgboost_dd_version_1` | Prototype workflow for reading ChEMBL 35 activity data, deriving molecular descriptors with RDKit, labelling compounds by a pIC50 threshold, and fitting an XGBoost classifier. |
| `ml_smilesfilter.py` | Earlier all-in-one experiment for data loading, descriptor calculation, model fitting, and candidate filtering. |
| `admet_screen.py` | SMILES screening through the pkCSM web service, with simple hepatotoxicity and hERG filters. |
| `lipin2.py` | Early TDP-43-focused exploration; includes 3D conformer generation and coordinate-based filtering. |

## Methods at a glance

The XGBoost prototype uses RDKit descriptors (molecular weight, LogP, TPSA, rotatable bonds, hydrogen-bond donors, and acceptors) as model features. Activity labels are derived from ChEMBL measurements using a pIC50 threshold. The goal is to explore a lightweight cheminformatics workflow from activity data to ranked molecules.

## Important limitations

- This is an early research prototype; scripts may need cleanup and adaptation before they run in a new environment.
- Model scores depend on the source data, assay selection, label definition, train/test split, and validation design. They are not experimental evidence.
- The coordinate-based filtering in the early scripts is a rough geometric heuristic, not molecular docking or binding-affinity prediction.
- ADMET predictions from an external service are computational estimates and require appropriate validation.
- Review each script’s inputs, dependencies, and assumptions before using its output.

## Getting started

The scripts expect Python plus scientific and cheminformatics packages such as RDKit, pandas, NumPy, scikit-learn, XGBoost, and tqdm. The ADMET script also uses requests and requires access to the pkCSM service.

The ChEMBL workflow reads a local ChEMBL 35 SQLite database named `chembl_35.db`; provide that database in the working directory before running it. Input/output filenames and paths are currently defined in the scripts.

Because these are exploratory scripts rather than a packaged command-line application, inspect the selected script and adjust paths and dependencies for your environment before running it.

## Research context

This repository reflects an exploration of machine learning and cheminformatics for molecular screening, alongside broader work on computational drug discovery. For a more complete project overview, see the [TDP-43 dissertation repository](https://github.com/aaryavashishth-beep/Masters).

## Author

**Aarya Vashishth** — Bioinformatics, computational drug discovery, and machine learning.
