import os
import argparse
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

import src.models.esmc_compat  # noqa: F401 - registers ESM-C HuggingFace loaders
from ..data.parsers import parse_dataset_file
from esm.models.esmc import ESMC
from esm.sdk import batch_executor
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient

load_dotenv()


def parse_fasta_sequences(file_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_id = None
    current_seq: List[str] = []
    annotation_chars = {"0", "1", "2", "-"}

    with open(file_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            elif current_id is not None:
                # CAID-style FASTA files may include annotation rows after sequence rows.
                if set(line).issubset(annotation_chars):
                    continue
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


def load_sequences(input_path: str, input_format: str) -> Dict[str, str]:
    path = Path(input_path)
    if input_format == "auto":
        input_format = "fasta" if path.suffix.lower() in {".fa", ".faa", ".fasta"} else "dataset"

    if input_format == "fasta":
        return parse_fasta_sequences(path)

    sequences, _, _, _ = parse_dataset_file(path)
    return sequences


class EmbeddingGenerator:
    """
    Generate and store protein embeddings using ESM-C models.

    HDF5 is the default output for portable submission use. Per-protein .pt files
    remain available for legacy research workflows.
    """

    def __init__(
        self,
        output: str,
        model_name: str = "esmc_600m",
        use_forge: bool = False,
        output_format: str = "h5",
    ):
        if output_format not in {"h5", "pt"}:
            raise ValueError("output_format must be 'h5' or 'pt'")

        self.output = Path(output)
        self.output_format = output_format
        self.model_name = model_name
        self.use_forge = use_forge
        self.token = os.getenv("ESM_KEY")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.output_format == "pt":
            self.output.mkdir(parents=True, exist_ok=True)
        else:
            self.output.parent.mkdir(parents=True, exist_ok=True)

        self.local_client = None
        self.forge_client = None
        self.failures: List[str] = []

        self._setup_model()

    def _setup_model(self) -> None:
        if self.use_forge:
            if not self.token:
                raise ValueError("ESM_KEY environment variable not found. Required for Forge API.")
            print(f"Setting up Forge Client (Model: {self.model_name})...")
            self.forge_client = ESM3ForgeInferenceClient(
                model=self.model_name,
                url="https://forge.evolutionaryscale.ai",
                token=self.token,
            )
        else:
            print(f"Loading local model {self.model_name} on {self.device}...")
            self.local_client = ESMC.from_pretrained(self.model_name).to(self.device).eval()

    def _embed_local(self, sequence: str):
        protein = ESMProtein(sequence=sequence)
        with torch.inference_mode():
            protein_tensor = self.local_client.encode(protein)
            return self.local_client.logits(
                protein_tensor,
                LogitsConfig(return_embeddings=True),
            )

    @staticmethod
    def _forge_worker(client, sequence):
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        if isinstance(protein_tensor, ESMProteinError):
            raise protein_tensor
        return client.logits(
            protein_tensor,
            LogitsConfig(return_embeddings=True),
        )

    def get_existing_ids(self) -> set[str]:
        if self.output_format == "pt":
            return {f.stem for f in self.output.glob("*.pt")}

        if not self.output.exists():
            return set()
        with h5py.File(self.output, "r") as handle:
            if "embeddings" not in handle:
                return set()
            return set(handle["embeddings"].keys())

    @staticmethod
    def _embedding_array(output) -> np.ndarray:
        embeddings = output.embeddings
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().float().numpy()
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.astype(np.float32, copy=False)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings

    def save_embedding(self, pid: str, sequence: str, output) -> None:
        embeddings = self._embedding_array(output)

        if self.output_format == "pt":
            data = {
                "pid": pid,
                "sequence": sequence,
                "embeddings": torch.from_numpy(embeddings),
            }
            torch.save(data, self.output / f"{pid}.pt")
            return

        with h5py.File(self.output, "a") as handle:
            group = handle.require_group("embeddings")
            if pid in group:
                del group[pid]
            dataset = group.create_dataset(pid, data=embeddings)
            dataset.attrs["sequence"] = sequence
            dataset.attrs["model_name"] = self.model_name

    def run(self, sequences: Dict[str, str]) -> List[str]:
        existing_ids = self.get_existing_ids()
        target_pids = [pid for pid in sequences.keys() if pid not in existing_ids]

        if not target_pids:
            print("All embeddings already exist.")
            return []

        print(f"Generating embeddings for {len(target_pids)} proteins...")

        if self.use_forge:
            self._run_forge(target_pids, sequences)
        else:
            self._run_local(target_pids, sequences)

        return self.failures

    def _run_local(self, pids, sequences) -> None:
        for pid in tqdm(pids, desc="Local Inference"):
            seq = sequences[pid]
            try:
                output = self._embed_local(seq)
                self.save_embedding(pid, seq, output)
            except Exception as exc:
                self.failures.append(pid)
                print(f"Error processing {pid}: {exc}")

    def _run_forge(self, pids, sequences) -> None:
        seqs = [sequences[pid] for pid in pids]

        print("Submitting batch to Forge API...")
        with batch_executor() as executor:
            results = executor.execute_batch(
                user_func=self._forge_worker,
                client=self.forge_client,
                sequence=seqs,
            )

            success_count = 0
            for pid, seq, result in zip(pids, seqs, results):
                if isinstance(result, Exception):
                    self.failures.append(pid)
                    print(f"Forge Error for {pid}: {result}")
                    continue

                self.save_embedding(pid, seq, result)
                success_count += 1

            print(f"Completed {success_count}/{len(pids)} successfully via Forge.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Protein Embeddings using ESM-C")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to a FASTA or cascDP dataset file containing protein sequences",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "dataset", "fasta"],
        default="auto",
        help="Input parser. 'auto' uses FASTA parsing for .fa/.faa/.fasta and dataset parsing otherwise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .h5 file when --output-format h5, or output directory when --output-format pt",
    )
    parser.add_argument(
        "--output-format",
        choices=["h5", "pt"],
        default="h5",
        help="Embedding output format. Default h5 writes one HDF5 file keyed by protein ID.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esmc_600m",
        help="Model name (e.g. esmc_300m, esmc_600m, esmc_6b)",
    )
    parser.add_argument("--forge", action="store_true", help="Use Forge API instead of local inference")

    args = parser.parse_args()

    print(f"Loading sequences from {args.dataset_file}...")
    try:
        sequences = load_sequences(args.dataset_file, args.input_format)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if not sequences:
        raise SystemExit("No sequences loaded.")

    print(f"Loaded {len(sequences)} sequences")

    generator = EmbeddingGenerator(
        output=args.output,
        output_format=args.output_format,
        model_name=args.model,
        use_forge=args.forge,
    )
    failures = generator.run(sequences)
    if failures:
        raise SystemExit(f"Failed to generate embeddings for {len(failures)} proteins: {failures[:10]}")
    print("Done.")


if __name__ == "__main__":
    main()
