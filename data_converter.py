"""Convert Agent 1 JSON output into a text knowledge base for Agent 2.

Agent 1 is assumed to have already extracted structured room data from a
floor plan. This module only normalizes that JSON into room-level text entries
that can be embedded and retrieved by the RAG pipeline.
"""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "sample_rooms.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "knowledge_base.txt"
ENTRY_SEPARATOR = "\n\n---\n\n"


class DataConversionError(ValueError):
    """Raised when Agent 1 JSON cannot be converted into room entries."""


def normalize_label(value: Any, default: str = "Unknown") -> str:
    """Convert JSON labels such as UTILITY into readable title-case text."""
    if value is None:
        return default

    label = str(value).replace("_", " ").strip()
    if not label:
        return default

    return label.title() if label.isupper() else label


def load_rooms(json_path: Path) -> list[dict[str, Any]]:
    """Load and validate room records from the Agent 1 JSON output."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Accept a single room object for convenience, but normalize to a list.
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise DataConversionError("Expected the JSON file to contain a list of rooms.")

    if not data:
        raise DataConversionError("Room JSON is empty. No knowledge base was created.")

    for index, room in enumerate(data, start=1):
        if not isinstance(room, dict):
            raise DataConversionError(f"Room record {index} is not a JSON object.")

    return data


def format_hazards(room: dict[str, Any]) -> list[str]:
    """Return readable hazard labels for a room, or an empty list if none exist."""
    hazards = room.get("hazards") or []

    if not isinstance(hazards, list):
        raise DataConversionError(
            f"Room {room.get('room_number', 'Unknown')} has a non-list hazards value."
        )

    return [
        normalize_label(hazard)
        for hazard in hazards
        if normalize_label(hazard, default="").strip()
    ]


def format_geometry(room: dict[str, Any]) -> list[str]:
    """Return the geometry lines that should be written for one room entry."""
    geometry = room.get("geometry")
    if not isinstance(geometry, dict) or not geometry:
        return ["None"]

    lines: list[str] = []
    area_cad_units = geometry.get("area_cad_units")
    if area_cad_units is not None:
        lines.append(f"Area CAD Units: {area_cad_units}")

    return lines or ["None"]


def room_to_text(room: dict[str, Any]) -> str:
    """Convert one room record into a structured natural-language entry."""
    room_number = normalize_label(room.get("room_number"))
    room_type = normalize_label(room.get("room_type"))
    zone_type = normalize_label(room.get("zone_type"))
    hazards = format_hazards(room)
    geometry_lines = format_geometry(room)

    lines = [
        f"Room Number: {room_number}",
        f"Room Type: {room_type}",
        f"Zone Type: {zone_type}",
        "Hazards:",
    ]
    lines.extend(hazards or ["None"])
    lines.extend(["", "Geometry:"])
    lines.extend(geometry_lines)

    return "\n".join(lines)


def convert_json_to_knowledge_base(
    json_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> str:
    """Read room JSON, convert it to text, and save the knowledge base file."""
    rooms = load_rooms(json_path)
    entries = [room_to_text(room) for room in rooms]
    knowledge_base = ENTRY_SEPARATOR.join(entries) + "\n"

    output_path.write_text(knowledge_base, encoding="utf-8")
    return knowledge_base


def main() -> None:
    """Run the JSON-to-knowledge-base conversion from the command line."""
    parser = argparse.ArgumentParser(
        description="Convert Agent 1 room JSON into knowledge_base.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the Agent 1 JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the knowledge base text file will be saved.",
    )
    args = parser.parse_args()

    try:
        convert_json_to_knowledge_base(args.input, args.output)
    except (DataConversionError, FileNotFoundError, JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Saved knowledge base to {args.output}")


if __name__ == "__main__":
    main()
