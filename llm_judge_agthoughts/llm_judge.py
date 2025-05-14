from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_api_key(key_file: Path | str) -> str:  # noqa: D401
    """Return the OpenAI API key stored in *key_file*."""
    key_path = Path(key_file).expanduser()
    with key_path.open() as fp:
        return json.load(fp)["gpt"]


def load_data(data_file: Path | str) -> List[Dict[str, Any]]:  # noqa: D401
    """Load the exported Label‑Studio JSON file as a list of dicts."""
    data_path = Path(data_file).expanduser()
    with data_path.open() as fp:
        return json.load(fp)


def flatten_metadata(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # noqa: D401
    """Remove annotator rows present in *raw*."""
    return [entry for entry in raw if entry["body"] != "Annotator"]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _initial_metadata_step(
    client: OpenAI,
    question: str,
    response: str,
    metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ask the LLM to convert *response* into structured metadata."""
    from llm_prompts import RESPONSE_TO_METADATA_COMPONENT

    prompt = RESPONSE_TO_METADATA_COMPONENT.format(question, response, metadata)

    raw = client.responses.create(model="gpt-4.1", input=prompt).output_text
    try:
        return ast.literal_eval(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("LLM returned non‑Python metadata") from exc


def _extract_annotation_flags(
    annotations: List[Dict[str, Any]],
) -> Tuple[bool | None, str | None, List[str] | None, List[str] | None]:
    """Pull annotator votes and comments from the Label‑Studio *annotations* block."""

    is_correct = next(
        (
            item["value"]["choices"][0]
            for item in annotations
            if item.get("from_name") == "label"
        ),
        None,
    )
    if is_correct is not None:
        is_correct = is_correct == "Yes"

    comment = next(
        (
            item["value"]["text"][0]
            for item in annotations
            if item.get("from_name") == "comment"
        ),
        None,
    )

    incorrect_meta_ids = next(
        (
            item["value"]["ranker"].get("flagged")
            for item in annotations
            if item.get("from_name") == "flag_items"
        ),
        None,
    )
    correct_meta_ids = next(
        (
            item["value"]["ranker"].get("unflagged")
            for item in annotations
            if item.get("from_name") == "flag_items"
        ),
        None,
    )

    return is_correct, comment, incorrect_meta_ids, correct_meta_ids


def _ids_to_entries(
    ids: List[str] | None, metadata: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if ids is None:
        return []
    return [item for item in metadata if item["id"] in ids]


def _apply_llm_corrections(
    client: OpenAI,
    question: str,
    response: str,
    incorrect_metadata: List[Dict[str, Any]],
    comment: str | None,
) -> List[Dict[str, Any]]:
    """Ask the model to provide corrections for *incorrect_metadata*."""
    from llm_prompts import incorrect_to_correct_prompt  # type: ignore

    prompt = incorrect_to_correct_prompt.format(question, response, incorrect_metadata, comment)
    raw = client.responses.create(
        model="gpt-4.1", input=prompt, temperature=0.2
    ).output_text

    if raw.strip() == "No correction found":
        return []
    try:
        corrected = ast.literal_eval(raw)
        for elem in corrected:
            elem["answer"] = elem["answer"].removeprefix("Correction: ")
        return corrected
    except Exception:  # noqa: BLE001
        return []


def _apply_llm_updates(
    client: OpenAI,
    question: str,
    response: str,
    corrected_metadata: List[Dict[str, Any]],
    comment: str | None,
) -> List[Dict[str, Any]]:
    """Ask the model to add missing metadata items or updates."""
    from llm_prompts import missing_correct_prompt  # type: ignore

    prompt = missing_correct_prompt.format(question, response, corrected_metadata, comment)
    raw = client.responses.create(model="gpt-4.1", input=prompt, temperature=0.2).output_text

    if raw.strip() == "No correction found":
        return []
    try:
        updated = ast.literal_eval(raw)
        for elem in updated:
            elem["answer"] = elem["answer"].removeprefix("Update: ")
        return updated
    except Exception:  # noqa: BLE001
        return []


def _flag_entries(
    correct_md: List[Dict[str, Any]],
    incorrect_md: List[Dict[str, Any]],
    corrected_md: List[Dict[str, Any]],
    updated_md: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine and flag all metadata buckets into the final structure."""
    # Start with explicit correct / incorrect buckets
    def add_flags(src: List[Dict[str, Any]], correct: bool) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for elem in src:
            clone = elem.copy()
            clone.update({
                "correct": correct,
                "correction": False,
                "update": False,
            })
            out.append(clone)
        return out

    final: List[Dict[str, Any]] = add_flags(correct_md, True) + add_flags(incorrect_md, False)

    # Merge corrections
    for patch in corrected_md:
        for target in final:
            if patch["id"] == target["id"] and patch["body"] == target["body"]:
                target["answer"] = patch["answer"]
                target["correction"] = True

    # Merge updates (can affect both buckets)
    for patch in updated_md:
        for target in final:
            if patch["id"] == target["id"] and patch["body"] == target["body"]:
                target["answer"] = patch["answer"]
                target["update"] = True

    return final


def get_final_metadata_with_flags(
    idx: int,
    data: List[Dict[str, Any]],
    client: OpenAI,
) -> List[Dict[str, Any]]:
    """High‑level wrapper that replicates the original notebook behaviour."""
    # ------------------------------------------------------------------
    # 1. Random‑like selection (if idx < 0 mimic original random logic)
    # ------------------------------------------------------------------
    if idx < 0:
        idx = random.randint(0, len(data) - 1)

    record = data[idx]
    question = record["data"]["processed_question"]
    response = record["data"]["answer"]
    raw_metadata = record["data"]["fields"]
    metadata = flatten_metadata(raw_metadata)

    # ------------------------------------------------------------------
    # 2. Initial metadata generation via LLM
    # ------------------------------------------------------------------
    metadata = _initial_metadata_step(client, question, response, metadata)

    # ------------------------------------------------------------------
    # 3. Annotation post‑processing
    # ------------------------------------------------------------------
    annotations = record["annotations"][0]["result"]
    _, comment, incorrect_ids, correct_ids = _extract_annotation_flags(annotations)

    incorrect_md = _ids_to_entries(incorrect_ids, metadata)
    correct_md = _ids_to_entries(correct_ids, metadata)

    # ------------------------------------------------------------------
    # 4. Corrections & updates
    # ------------------------------------------------------------------
    corrected_md = _apply_llm_corrections(
        client, question, response, incorrect_md, comment
    )
    all_corrected_md = correct_md + corrected_md

    updated_md = _apply_llm_updates(
        client, question, response, all_corrected_md, comment
    )

    # ------------------------------------------------------------------
    # 5. Combine & flag
    # ------------------------------------------------------------------
    return _flag_entries(correct_md, incorrect_md, corrected_md, updated_md)


def run(index):
    data = load_data("/work/mech-ai-scratch/shreyang/Agentic Ag/label-studio/annotation-176.json")
    client = OpenAI(api_key=load_api_key("keys.json"))
    result = get_final_metadata_with_flags(index, data, client)
    return result

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Produce final_metadata_with_flags for a given index")
    parser.add_argument("--data-file", required=True, help="Path to the Label‑Studio JSON export")
    parser.add_argument("--key-file", default="keys.json", help="Path to keys.json containing the GPT key")
    parser.add_argument("--index", type=int, default=-1, help="Annotation index to process (negative = random)")
    args = parser.parse_args()

    client = OpenAI(api_key=load_api_key(args.key_file))
    data = load_data(args.data_file)

    result = get_final_metadata_with_flags(args.index, data, client)
    print(result)


if __name__ == "__main__":  # pragma: no cover
    _cli()
