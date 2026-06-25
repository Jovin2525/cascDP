"""
Step 1 of PDB_missing pipeline.

Fetches all X-ray polymer entity IDs from the RCSB PDB Search API and saves
them to a text file, one ID per line.

Usage:
    python scripts/data/fetch_pdb_entity_ids.py [--output data/pdb_entity_ids.txt]

Output:
    A plain text file with ~231K lines of the form: 4HHB_1
"""

import argparse
import requests
from pathlib import Path

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
SEARCH_PAYLOAD = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": "X-RAY DIFFRACTION"
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match",
                    "value": "Protein"
                }
            }
        ]
    },
    "request_options": {
        "return_all_hits": True
    },
    "return_type": "polymer_entity"
}

def fetch_entity_ids(output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Querying RCSB Search API for X-ray protein polymer entities...")
    response = requests.post(SEARCH_URL, json=SEARCH_PAYLOAD, timeout=120)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        raise RuntimeError(f"RCSB Search API returned {response.status_code}")

    data = response.json()
    entity_ids = [hit["identifier"] for hit in data.get("result_set", [])]

    with open(output_path, "w") as f:
        for eid in entity_ids:
            f.write(f"{eid}\n")

    print(f"Saved {len(entity_ids):,} entity IDs → {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fetch PDB polymer entity IDs")
    parser.add_argument("--output", default="data/pdb_entity_ids.txt",
                        help="Output text file path (default: data/pdb_entity_ids.txt)")
    args = parser.parse_args()
    fetch_entity_ids(args.output)

if __name__ == "__main__":
    main()