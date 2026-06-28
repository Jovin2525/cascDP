# cascDP

cascDP is a cascaded protein residue predictor for intrinsic disorder, binding, and linker regions. The project contains the full workflow for data preparation, training, evaluation, ablation experiments, and the offline 2026 CAID submission runtime.

## CAID Submission

The CAID submission runtime is a CPU-only Docker image that reads a FASTA file, precomputed ESM-C embeddings, and a cascDP checkpoint, then writes per-protein CAID `.caid` files under task-specific flavor directories. PLM weights, checkpoints, and embeddings are external runtime inputs.

Quick start (Docker):

```bash
docker build -t cascdp-caid4 .
docker run --rm \
    -v /absolute/path/to/cascDP/data/input.fasta:/work/input.fasta:ro \
    -v /absolute/path/to/cascDP/data/embeddings.h5:/work/embeddings.h5:ro \
    -v /absolute/path/to/cascDP/checkpoints/cascDP.pt:/work/model.pt:ro \
    -v /absolute/path/to/cascDP/data/submission_out:/work/submission \
    cascdp-caid4 \
    --checkpoint /work/model.pt --fasta /work/input.fasta \
    --embeddings /work/embeddings.h5 --output-dir /work/submission \
    --output-prefix cascDP --tasks disorder,binding,linker --threads 24
```

Replace `/absolute/path/to/cascDP` with the absolute path to your local repository checkout.

For full submission instructions — embedding generation, input formats, local runs, per-protein embedding directories, and submission notes — see **[SUBMISSION.md](SUBMISSION.md)**.

## What This Repository Contains - Pending Changes

- Phase 1 disorder prediction models built on ESM-C backbones.
- Phase 2 binding and linker heads cascaded on top of the Phase 1 disorder model.
- Separate, joint, ablation, and MLP-cascade Phase 2 experiment paths.
- Dataset construction scripts for DisProt, PDB_missing, CAID filtering, and Phase 2 filtering.
- Internal and CAID-official evaluation CLIs.
- A CPU-only Docker submission runtime that reads FASTA plus precomputed embeddings and writes CAID `.caid` files.

## Repository Layout

```text
configs/                         Training configs for main cascDP models
data/                            Local datasets, CAID references, caches, and embeddings
scripts/data/run_pipeline.sh     End-to-end data preparation pipeline
src/cli/                         Training, prediction, evaluation, plotting, and utility CLIs
src/data/                        Dataset loaders, parsers, and preprocessing modules
src/embeddings/                  Embedding generation utilities
src/evaluation/                  Metrics, CAID writers, and CAID sidecar integration
src/experiments/                 Ablation and MLP-cascade experiment variants
src/models/                      Backbone, context, fusion, Phase 1, and Phase 2 models
src/training/                    Losses and training loop implementation
third_party/caid/                Vendored CAID assessment code
Dockerfile                       CAID submission runtime image
SUBMISSION.md                    CAID Docker submission runtime instructions
environment.yml                  Canonical full-project conda environment
requirements.txt                 Python dependencies for the full project
requirements-submission.txt      Minimal Python dependency layer for CAID submission runtime
```

Large runtime artifacts such as checkpoints, embeddings, PLM weights, and external databases are expected to be provided locally. They are not bundled into the Docker image.

## Environment Setup

Use the conda environment for full project workflows. It installs both Python dependencies and `mmseqs2`, which is required by preprocessing and cannot be represented by `requirements.txt` alone.

```bash
conda env create -f environment.yml
conda activate cascDP
pip install --no-deps "esm@git+https://github.com/Biohub/esm.git@82ee35553d39169d678f784c8d3f8712ffd7d2c4"
```

`requirements.txt` includes the runtime packages that ESM-C imports. Install `esm` with `--no-deps` so pip does not replace the pinned HuggingFace `transformers==4.48.1` with Biohub's transformers fork or install unused structure-processing/visualization packages such as `rdkit`, `pydssp`, and `py3dmol`.

To sync an existing environment back to the repository spec:

```bash
conda env update -f environment.yml --prune
conda activate cascDP
```

For the CAID Docker submission runtime, see [SUBMISSION.md](SUBMISSION.md).

## Data Pipeline

The full data preparation pipeline is wrapped by `scripts/data/run_pipeline.sh`. The script changes to the repository root automatically and chains these steps:

1. Fetch PDB entity IDs.
2. Build the clustered PDB_missing dataset.
3. Merge DisProt with PDB_missing, filter CAID3 exact matches, cluster the master pool, and create train/val/test splits.
4. Filter splits for Phase 2 binding/linker supervision.

Run the full pipeline:

```bash
bash scripts/data/run_pipeline.sh
```

Common variants:

```bash
# Reuse existing entity IDs, then rebuild and refilter.
bash scripts/data/run_pipeline.sh --skip-fetch

# Only run later steps with a specific Python executable.
bash scripts/data/run_pipeline.sh --skip-fetch --skip-build --python /path/to/envs/cascDP/bin/python

# Only run the Phase 2 filter on existing splits.
bash scripts/data/run_pipeline.sh --skip-fetch --skip-build --skip-merge
```

Useful individual preprocessing entry points:

```bash
python -m src.data.preprocessing.fetch_pdb_entity_ids --output data/pdb_entity_ids.txt
python -m src.data.preprocessing.create_pdb_missing_dataset --ids data/pdb_entity_ids.txt --output data/final_cleaned_dataset/pdb_missing_caid4.txt
python -m src.data.preprocessing.create_dataset_new_caid4
python -m src.data.preprocessing.filter_phase2_dataset --input_dir data/final_cleaned_dataset --prefix final_update_or_caid4 --splits train val test
```

Primary dataset files used by the default configs include:

- `data/final_cleaned_dataset/train_final_update_or_caid4_unaltered_data.txt`
- `data/final_cleaned_dataset/val_final_update_or_caid4_unaltered_data.txt`
- `data/final_cleaned_dataset/test_final_update_or_caid4_unaltered_data.txt`
- `data/final_cleaned_dataset/train_final_update_or_caid4_phase2_any_data.txt`
- `data/final_cleaned_dataset/val_final_update_or_caid4_phase2_any_data.txt`
- `data/final_cleaned_dataset/test_final_update_or_caid4_phase2_any_data.txt`

## Training

All main training configs live under `configs/`. Most current configs use `embedding_mode: on_the_fly`, so the model creates embeddings during training through the configured ESM-C backbone. If you switch a config to precomputed embeddings, provide matching `train_embedding_dir` and `val_embedding_dir` values. See [SUBMISSION.md](SUBMISSION.md#embedding-generation) for the embedding generation guide.

### Phase 1 Disorder

```bash
python -m src.cli.train --config configs/phase1/disorder.yaml
```

Other Phase 1 configs:

```bash
python -m src.cli.train --config configs/phase1/disorder_300m.yaml
python -m src.cli.train --config configs/phase1/disorder_gate.yaml
python -m src.cli.train --config configs/phase1/disprot_only.yaml
python -m src.cli.train --config configs/phase1/pdb_only.yaml
```

Recycle variant:

```bash
python -m src.cli.train_recycle --config configs/phase1/recycle.yaml
```

### Phase 2 Binding And Linker

The primary Phase 2 strategy trains binding and linker heads separately on top of a compatible Phase 1 checkpoint, then merges them into one checkpoint.

```bash
python -m src.cli.train --config configs/phase2/binding.yaml
python -m src.cli.train --config configs/phase2/linker.yaml
```

Merge the separately trained heads:

```bash
python -m src.cli.merge_phase2 \
    --binding_ckpt checkpoints/phase2/binding/best_model.pt \
    --linker_ckpt checkpoints/phase2/linker/best_model.pt \
    --output checkpoints/cascDP.pt
```

Joint Phase 2 training is also available:

```bash
python -m src.cli.train --config configs/phase2/joint.yaml
```

### Experimental Variants

Ablation without disorder cascade:

```bash
python -m src.cli.train_ablation_no_cascade --config configs/experiments/ablation_no_cascade/binding.yaml
python -m src.cli.train_ablation_no_cascade --config configs/experiments/ablation_no_cascade/linker.yaml
python -m src.cli.train_ablation_no_cascade --config configs/experiments/ablation_no_cascade/binding_linker.yaml
```

MLP-cascade Phase 2 variant - pending removal:

```bash
python -m src.cli.train_phase2_mlp_cascade --config configs/experiments/phase2_mlp_cascade/binding.yaml
python -m src.cli.train_phase2_mlp_cascade --config configs/experiments/phase2_mlp_cascade/linker.yaml
```

Training CLIs also support `--resume` and `--log-level` where implemented. Select GPUs with `CUDA_VISIBLE_DEVICES` rather than a CLI flag.

## Evaluation

Use `src.cli.evaluate_caid` for CAID reporting. CAID metrics are computed via the vendored `third_party/caid/` package ([BioComputingUP/CAID](https://github.com/BioComputingUP/CAID)).

Supported `evaluate_caid` test sets:

- `test_final` - independent test set constructed from DisProt and RCSB PDB
- `caid3_disorder_nox`
- `caid3_disorder_pdb`
- `caid3_binding`
- `caid3_binding_idr`
- `caid3_linker`

Examples:

```bash
# CAID evaluation output.
python -m src.cli.evaluate_caid \
    --checkpoint checkpoints/cascDP.pt \
    --test-set caid3_binding \
    --output-dir results/evaluations/cascdp

# Faster local check without bootstrap metrics.
python -m src.cli.evaluate_caid \
    --checkpoint checkpoints/cascDP.pt \
    --test-set caid3_linker \
    --output-dir results/evaluations/cascdp \
    --skip-bootstrap

# Evaluate only selected output flavors.
python -m src.cli.evaluate_caid \
    --checkpoint checkpoints/cascDP.pt \
    --test-set caid3_binding \
    --flavors binding
```

For multi-label binding checkpoints, select a per-type head instead of the combined noisy-OR score:

```bash
python -m src.cli.evaluate_caid \
    --checkpoint checkpoints/cascDP.pt \
    --test-set caid3_binding \
    --binding-head protein
```

Evaluate experiment variants:

```bash
python -m src.cli.evaluate_ablation_no_cascade \
    --checkpoint checkpoints/ablation_no_cascade/binding/best_model.pt \
    --test-set caid3_binding \
    --output-dir results/evaluations/ablation_no_cascade

python -m src.cli.evaluate_phase2_mlp_cascade \
    --checkpoint checkpoints/phase2_mlp_cascade/binding/best_model.pt \
    --test-set caid3_binding \
    --output-dir results/evaluations/phase2_mlp_cascade
```

Evaluation CLIs write per-test-set CAID submissions under `<output-dir>/submissions/<test-set>/` and metrics under `<output-dir>/caid_metrics/<test-set>/`.

## Prediction For Research Use - In Progress

For general local prediction from FASTA, use `src.cli.predict`. This path builds the model/backbone from a training config and can optionally write plots.

```bash
python -m src.cli.predict \
    --checkpoint checkpoints/cascDP.pt \
    --config configs/phase2/joint.yaml \
    --fasta input.fasta \
    --output-dir results/predictions \
    --plot
```

Use the CAID Docker submission runtime (see [SUBMISSION.md](SUBMISSION.md)) when you need assessor-facing `.caid` output from precomputed embeddings.

## Attribution and Citations

This repository incorporates code from the Critical Assessment of Protein Intrinsic Disorder Prediction (CAID) community, licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.

If you use this software, please cite the official CAID publications:

- **CAID2:** Conte AD, Mehdiabadi M, Bouhraoua A, Miguel Monzon A, Tosatto SCE, Piovesan D. Critical assessment of protein intrinsic disorder prediction (CAID) - Results of round 2. *Proteins*. 2023; 91(12): 1925-1934. https://doi.org/10.1002/prot.26582
- **CAID1:** Necci M, Piovesan D, CAID Predictors, et al. Critical assessment of protein intrinsic disorder prediction. *Nat Methods*. 2021; 18: 472-481. https://doi.org/10.1038/s41592-021-01117-3

The vendored CAID evaluation code is located under `third_party/caid/`. Project integration code adapts cascDP predictions to CAID-compatible input and output formats.

## To Note

- Use `environment.yml`, not only `requirements.txt`, when preprocessing requires `mmseqs2`.
- Keep FASTA IDs, embedding IDs, and checkpoint protein IDs consistent.
- Ensure embedding hidden size matches the checkpoint backbone. The default ESM-C 600M configs expect hidden size `1152`.
- Phase 2 configs require a Phase 1 checkpoint with matching context/backbone settings.
