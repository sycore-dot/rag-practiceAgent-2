"""CLI and Python API for Agent 2 Fire Safety Code Agent."""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from build_vectorstore import build_vectorstore, vectorstore_ready
from rag_chain import FireSafetyCodeRAG, build_code_rag, validate_room_output_schema
from settings import INPUT_ROOMS_PATH


class CodeAgentInputError(ValueError):
    """Raised when Agent 1 room JSON is missing or malformed."""


def validate_rooms(data: Any) -> list[dict[str, Any]]:
    """Validate Agent 1 output and normalize a single object to a list."""
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise CodeAgentInputError("Expected room JSON to be a list of objects.")
    if not data:
        raise CodeAgentInputError("Room JSON is empty.")

    rooms: list[dict[str, Any]] = []
    for index, room in enumerate(data, start=1):
        if not isinstance(room, dict):
            raise CodeAgentInputError(f"Room record {index} is not a JSON object.")
        rooms.append(room)

    return rooms


def load_rooms_from_file(path: Path = INPUT_ROOMS_PATH) -> list[dict[str, Any]]:
    """Load Agent 1 room JSON from a file."""
    if not path.exists():
        raise FileNotFoundError(f"Input room JSON not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as error:
        raise CodeAgentInputError(f"Invalid JSON in {path}: {error}") from error

    return validate_rooms(data)


def load_rooms_from_text(text: str) -> list[dict[str, Any]]:
    """Load Agent 1 room JSON from a Streamlit text area."""
    try:
        data = json.loads(text)
    except JSONDecodeError as error:
        raise CodeAgentInputError(f"Invalid room JSON: {error}") from error

    return validate_rooms(data)


def ensure_vectorstore(auto_build: bool = True) -> None:
    """Build Chroma automatically when the rule vector store is missing."""
    if vectorstore_ready():
        return
    if not auto_build:
        raise FileNotFoundError("Chroma vector store is missing. Run build_vectorstore.py.")
    build_vectorstore()


def analyze_rooms(
    rooms: Any,
    *,
    agent: FireSafetyCodeRAG | None = None,
    auto_build: bool = True,
) -> list[dict[str, Any]]:
    """Analyze one or more rooms and return a strict JSON list."""
    validated_rooms = validate_rooms(rooms)
    ensure_vectorstore(auto_build=auto_build)
    active_agent = agent or build_code_rag()
    results = active_agent.analyze_rooms(validated_rooms)
    for result in results:
        validate_room_output_schema(result)
    return results


def main() -> None:
    """Run Agent 2 from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze Agent 1 room JSON with the Fire Safety Code Agent."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_ROOMS_PATH,
        help="Path to Agent 1 room JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the strict Agent 2 output JSON.",
    )
    parser.add_argument(
        "--no-auto-build",
        action="store_true",
        help="Fail instead of building Chroma when the vector store is missing.",
    )
    args = parser.parse_args()

    rooms = load_rooms_from_file(args.input)
    result = analyze_rooms(rooms, auto_build=not args.no_auto_build)
    output_text = json.dumps(result, indent=2)

    if args.output:
        args.output.write_text(output_text + "\n", encoding="utf-8")

    print(output_text)


if __name__ == "__main__":
    main()
