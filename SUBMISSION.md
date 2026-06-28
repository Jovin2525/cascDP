# CAID Docker Submission Runtime

CAID submission path is prediction only. It reads a FASTA file, [precomputed embeddings](#embedding-generation), and a cascDP checkpoint, then writes per-protein CAID output files under task-specific flavor directories.

The Docker image defined by `Dockerfile` is the canonical submission package. The image is CPU-only by design and installs the minimal dependency layer from `requirements-submission.txt` plus CPU Torch. It does not include protein language model weights. Embeddings, checkpoints, and input FASTA files are mounted as external runtime inputs.

## Submission Inputs

- FASTA file with one or more protein sequences.
- Embedding source passed with `--embeddings`: either a single `.h5`/`.hdf5` container keyed by protein ID, or a directory of per-protein `.pt`, `.npy`, `.h5`, or `.hdf5` files named `{protein_id}.{suffix}` (see [Embedding Generation](#embedding-generation) if you need to produce these).
- cascDP checkpoint compatible with the embedding backbone and hidden size.

A single multi-protein `.npy` file is not supported because plain NPY arrays do not reliably carry protein IDs. For one-file assessor-provided embeddings, use HDF5; for NPY, provide one file per protein in a directory.

FASTA IDs and embedding IDs must match exactly. Checkpoints and embeddings must also agree on backbone configuration and hidden size. The parser accepts plain FASTA; it also tolerates CAID-style files that interleave annotation rows despite a `.fasta` suffix.

## Docker Submission Run

Build the image:

```bash
docker build -t cascdp-caid4 .
```

Run with a single HDF5 embeddings container. This is the canonical layout produced by `src.embeddings.generate_embeddings`:

Replace `/absolute/path/to/cascDP` with the absolute path to your local repository checkout.

```bash
docker run --rm \
    -v /absolute/path/to/cascDP/data/input.fasta:/work/input.fasta:ro \
    -v /absolute/path/to/cascDP/data/embeddings.h5:/work/embeddings.h5:ro \
    -v /absolute/path/to/cascDP/checkpoints/cascDP.pt:/work/model.pt:ro \
    -v /absolute/path/to/cascDP/data/submission_out:/work/submission \
    cascdp-caid4 \
    --checkpoint /work/model.pt \
    --fasta /work/input.fasta \
    --embeddings /work/embeddings.h5 \
    --output-dir /work/submission \
    --output-prefix cascDP \
    --tasks disorder,binding,linker \
    --threads 24
```

If embeddings are provided as per-protein files, mount the embedding directory instead. The file name stem must match the FASTA protein ID, for example `P04637.h5` for `>P04637`:

```bash
docker run --rm \
    -v /absolute/path/to/cascDP/data/input.fasta:/work/input.fasta:ro \
    -v /absolute/path/to/cascDP/data/embeddings:/work/embeddings:ro \
    -v /absolute/path/to/cascDP/checkpoints/cascDP.pt:/work/model.pt:ro \
    -v /absolute/path/to/cascDP/data/submission_out:/work/submission \
    cascdp-caid4 \
    --checkpoint /work/model.pt \
    --fasta /work/input.fasta \
    --embeddings /work/embeddings \
    --output-dir /work/submission \
    --output-prefix cascDP \
    --tasks disorder,binding,linker \
    --threads 24
```

Per-protein embedding directories may contain `.npy`, `.h5`, `.hdf5`, or `.pt` files named `{protein_id}.{suffix}`. A single multi-protein `.npy` file is not supported because plain NPY arrays do not reliably carry protein IDs.

## Local Submission Run

To run predictions locally outside Docker, run the same predictor directly in the full project environment:

```bash
conda activate cascDP
python -m src.cli.predict_submission \
    --checkpoint checkpoints/phase2/unified_esm_unfrozen_v3_nolora.pt \
    --fasta /path/to/input.fasta \
    --embeddings /path/to/embeddings.h5 \
    --output-dir /path/to/submission_out \
    --output-prefix cascDP \
    --tasks disorder,binding,linker \
    --threads 24
```

## Submission Outputs

For the Docker examples above, `<output-dir>` is the mounted host directory `/absolute/path/to/cascDP/data/submission_out`. For local runs, it is the directory passed to `--output-dir`.

Both Docker and local runs write one file per protein under each requested task flavor, plus `timings.csv`:

- `<output-dir>/disorder/{protein_id}.caid`
- `<output-dir>/binding/{protein_id}.caid`
- `<output-dir>/linker/{protein_id}.caid`
- `<output-dir>/timings.csv`

Output rows use CAID format:

```text
>protein_id
1	M	0.892000	1
2	E	0.813000	1
```

Ambiguous residues such as `B`, `Z`, `J`, `U`, `O`, and `X` are preserved in output rows and receive scores like other residues.

`timings.csv` is written once per predictor invocation and records one elapsed time per input protein. If several tasks are requested in the same run, the same timings apply to all output flavors.

## Submission Notes

- The runtime is CPU-only and limits `--threads` to 1 through 24.
- PLM weights are not bundled in the container; checkpoints, embeddings, and databases are external runtime inputs.
- Use `--tasks disorder,binding,linker` for the unified model submission. The run fails if any requested task is not produced.
- Use `--tasks all` only for exploratory runs where writing every available checkpoint output is acceptable.
- Saved finite thresholds in the checkpoint are required to write binary CAID states. Tasks with absent or non-finite thresholds are treated as missing for explicit `--tasks` requests.

## Embedding Generation

This section is only needed if you do not already have precomputed embeddings.

Before generating embeddings locally, create and activate the full project environment using the [README environment setup](README.md#environment-setup).

Generate one HDF5 embedding file by default:

```bash
python -m src.embeddings.generate_embeddings \
    --dataset_file data/final_cleaned_dataset/train_final_update_or_caid4_unaltered_data.txt \
    --output data/embeddings/esmc_600m/train.h5 \
    --model esmc_600m

python -m src.embeddings.generate_embeddings \
    --dataset_file data/final_cleaned_dataset/val_final_update_or_caid4_unaltered_data.txt \
    --output data/embeddings/esmc_600m/val.h5 \
    --model esmc_600m
```

The generator also accepts plain FASTA input. Use `--input-format fasta` when you need exact FASTA header IDs preserved, especially for CAID-style target files. Dataset files with source metadata should use the default dataset parser; headers like `protein_id|source` are parsed as protein ID plus source.

Additionally, per-protein `.pt` output is available for research workflows:

```bash
python -m src.embeddings.generate_embeddings \
    --dataset_file data/final_cleaned_dataset/train_final_update_or_caid4_unaltered_data.txt \
    --output data/embeddings/esmc_600m/train \
    --output-format pt \
    --model esmc_600m
```

The HDF5 layout is keyed by protein ID and is directly accepted by training, evaluation, and Docker submission loaders:

```text
embeddings.h5
└── embeddings/
    ├── protein_id_1  # dataset, shape (L, hidden_dim)
    └── protein_id_2
```

---

See [README.md](README.md) for the general research workflow (data preparation, training, evaluation, ablations).
