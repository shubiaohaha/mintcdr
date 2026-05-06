from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mintcdr.prompts import PromptTemplate, build_item_utility_prompt
from mintcdr.semantic import OfflineSemanticEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic semantic feature JSON files from metadata CSV files.")
    parser.add_argument("--items", required=True, help="CSV: domain,item_id,title,category,description,attributes")
    parser.add_argument("--users", required=True, help="CSV: user_id,profile,behavior_sequence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--source-domain-desc", default="source domain")
    parser.add_argument("--target-domain-desc", default="target domain")
    args = parser.parse_args()

    encoder = OfflineSemanticEncoder(args.semantic_dim)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    item_template = PromptTemplate.from_file("prompts/item_semantic_prompt.txt")
    user_template = PromptTemplate.from_file("prompts/user_semantic_prompt.txt")
    item_sem = {}
    utility_prompts = {}

    with Path(args.items).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prompt = item_template.render(
                item_title=row.get("title", ""),
                item_category=row.get("category", ""),
                item_description=row.get("description", ""),
            )
            key = f"{row['domain']}:{row['item_id']}"
            item_sem[key] = encoder.encode(prompt).tolist()
            if row["domain"] == "source":
                utility_prompts[key] = build_item_utility_prompt(
                    item_description=row.get("description", ""),
                    item_attributes=row.get("attributes", ""),
                    source_domain_description=args.source_domain_desc,
                    target_domain_description=args.target_domain_desc,
                )

    user_sem = {}
    with Path(args.users).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prompt = user_template.render(
                user_profile=row.get("profile", ""),
                behavior_sequence=row.get("behavior_sequence", ""),
            )
            user_sem[row["user_id"]] = encoder.encode(prompt).tolist()

    with (output / "item_semantics.json").open("w", encoding="utf-8") as f:
        json.dump(item_sem, f, ensure_ascii=False)
    with (output / "user_semantics.json").open("w", encoding="utf-8") as f:
        json.dump(user_sem, f, ensure_ascii=False)
    with (output / "source_utility_prompts.json").open("w", encoding="utf-8") as f:
        json.dump(utility_prompts, f, ensure_ascii=False, indent=2)
    print(f"wrote semantic features to {output}")


if __name__ == "__main__":
    main()

