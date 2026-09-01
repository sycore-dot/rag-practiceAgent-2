"""RAG chain for Agent 2 fire-requirement code generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CODE_AGENT_VERSION,
    ProjectConfigurationError,
    ensure_gemini_api_key,
    get_chat_model,
    get_embedding_model,
)


class RAGConfigurationError(RuntimeError):
    """Raised when the RAG pipeline cannot run safely."""


SYSTEM_PROMPT = """You are a Senior Fire Protection Engineer with 10 years of experience in commercial and industrial fire safety systems.

Use only retrieved NBC code sections.
Never hallucinate.
Provide confidence and reasoning.
Return only valid JSON.

Rules:
- Treat retrieved context as the only source of code requirements.
- Apply a retrieved rule only when it explicitly matches the room type, category, or hazards.
- If a field is not specified in the retrieved NBC context, use false, null, "Not specified", or [] according to the schema.
- Do not invent NBC clauses, hazard classes, detector types, suppression systems, or assumptions.
- Return a single JSON object and no markdown."""


HUMAN_PROMPT = """Analyze this Agent 1 room JSON:
{room_json}

Retrieved NBC code sections:
{retrieved_context}

Return JSON using exactly this schema:
{schema_instructions}"""


SCHEMA_INSTRUCTIONS = json.dumps(
    {
        "room_number": "same as input",
        "room_type": "same as input",
        "area_sqft": "same as input or null",
        "capacity": "same as input or null",
        "ceiling_height_ft": "same as input or null",
        "zone_type": "same as input",
        "hazards": "same as input list",
        "geometry": "same as input or null",
        "code_agent_version": CODE_AGENT_VERSION,
        "fire_requirements": {
            "mapped_category": "string",
            "hazard_class": "string",
            "sprinkler_required": True,
            "sprinkler_system_type": "string or Not specified",
            "detector_required": True,
            "detector_type": "string or Not specified",
            "manual_call_point_required": False,
            "extinguisher_required": True,
            "special_suppression_required": False,
            "special_suppression_type": None,
            "classification_confidence": "HIGH, MEDIUM, or LOW",
            "confidence_reason": "string",
            "rule_type_used": ["NBC"],
            "review_required": False,
            "review_reason": None,
            "assumptions": [],
            "rule_references": [],
        },
    },
    indent=2,
)


FIRE_REQUIREMENT_KEYS = [
    "mapped_category",
    "hazard_class",
    "sprinkler_required",
    "sprinkler_system_type",
    "detector_required",
    "detector_type",
    "manual_call_point_required",
    "extinguisher_required",
    "special_suppression_required",
    "special_suppression_type",
    "classification_confidence",
    "confidence_reason",
    "rule_type_used",
    "review_required",
    "review_reason",
    "assumptions",
    "rule_references",
]


ROOM_OUTPUT_KEYS = [
    "room_number",
    "room_type",
    "area_sqft",
    "capacity",
    "ceiling_height_ft",
    "zone_type",
    "hazards",
    "geometry",
    "code_agent_version",
    "fire_requirements",
]


def get_langchain_dependencies() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Import LangChain dependencies lazily for readable errors."""
    try:
        from langchain_chroma import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
    except ModuleNotFoundError as error:
        raise RAGConfigurationError(
            "Missing LangChain, Chroma, or Gemini dependency. "
            "Run: pip install -r requirements.txt"
        ) from error

    return Chroma, ChatPromptTemplate, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def load_vectorstore(vectorstore_dir: Path = CHROMA_DIR) -> Any:
    """Load a persistent Chroma vector store."""
    ensure_gemini_api_key()
    Chroma, _, _, GoogleGenerativeAIEmbeddings = get_langchain_dependencies()

    metadata_path = vectorstore_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Chroma vector store not found at {vectorstore_dir}. "
            "Run: python build_vectorstore.py"
        )

    embeddings = GoogleGenerativeAIEmbeddings(model=get_embedding_model(True))
    return Chroma(
        persist_directory=str(vectorstore_dir),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )


def message_content_to_text(content: Any) -> str:
    """Normalize LangChain response content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise RAGConfigurationError("Gemini did not return a JSON object.")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise RAGConfigurationError(
                "Gemini returned malformed JSON. Re-run the request."
            ) from error

    if not isinstance(parsed, dict):
        raise RAGConfigurationError("Gemini returned JSON, but not a JSON object.")

    return parsed


def normalize_text(value: Any, default: str = "Not specified") -> str:
    """Normalize optional strings for the output schema."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_bool(value: Any) -> bool:
    """Normalize booleans returned by the LLM."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "required", "1"}
    return bool(value)


def normalize_list(value: Any) -> list[Any]:
    """Normalize list-like values returned by the LLM."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def default_fire_requirements() -> dict[str, Any]:
    """Return the exact nested fire_requirements schema with safe defaults."""
    return {
        "mapped_category": "Not specified",
        "hazard_class": "Not specified",
        "sprinkler_required": False,
        "sprinkler_system_type": "Not specified",
        "detector_required": False,
        "detector_type": "Not specified",
        "manual_call_point_required": False,
        "extinguisher_required": False,
        "special_suppression_required": False,
        "special_suppression_type": None,
        "classification_confidence": "LOW",
        "confidence_reason": "",
        "rule_type_used": [],
        "review_required": True,
        "review_reason": "No retrieved NBC rule explicitly covered this room.",
        "assumptions": [],
        "rule_references": [],
    }


def references_from_documents(documents: list[Any]) -> list[str]:
    """Extract unique NBC references from retrieved rule chunks."""
    references: list[str] = []
    for document in documents:
        reference = str(document.metadata.get("reference", "")).strip()
        if reference and reference != "Not specified" and reference not in references:
            references.append(reference)
    return references


def normalize_fire_requirements(
    llm_payload: dict[str, Any],
    retrieved_documents: list[Any],
) -> dict[str, Any]:
    """Coerce LLM output into the exact fire_requirements schema."""
    source = llm_payload.get("fire_requirements", llm_payload)
    if not isinstance(source, dict):
        source = {}

    result = default_fire_requirements()
    result.update({key: source.get(key, result[key]) for key in FIRE_REQUIREMENT_KEYS})

    result["mapped_category"] = normalize_text(result["mapped_category"])
    result["hazard_class"] = normalize_text(result["hazard_class"])
    result["sprinkler_required"] = normalize_bool(result["sprinkler_required"])
    result["sprinkler_system_type"] = normalize_text(result["sprinkler_system_type"])
    result["detector_required"] = normalize_bool(result["detector_required"])
    result["detector_type"] = normalize_text(result["detector_type"])
    result["manual_call_point_required"] = normalize_bool(
        result["manual_call_point_required"]
    )
    result["extinguisher_required"] = normalize_bool(result["extinguisher_required"])
    result["special_suppression_required"] = normalize_bool(
        result["special_suppression_required"]
    )
    if not result["special_suppression_required"]:
        result["special_suppression_type"] = None
    elif result["special_suppression_type"] is not None:
        result["special_suppression_type"] = normalize_text(
            result["special_suppression_type"]
        )

    confidence = normalize_text(result["classification_confidence"], default="LOW")
    confidence = confidence.upper()
    result["classification_confidence"] = (
        confidence if confidence in {"HIGH", "MEDIUM", "LOW"} else "LOW"
    )

    references = references_from_documents(retrieved_documents)
    allowed_references = set(references)
    llm_references = [
        str(item).strip()
        for item in normalize_list(result["rule_references"])
        if str(item).strip() in allowed_references
    ]
    result["rule_references"] = [
        reference
        for reference in [*llm_references, *references]
        if reference and reference != "Not specified"
    ]
    result["rule_references"] = list(dict.fromkeys(result["rule_references"]))

    result["rule_type_used"] = [
        str(item).strip()
        for item in normalize_list(result["rule_type_used"])
        if str(item).strip()
    ]
    if result["rule_references"] and "NBC" not in result["rule_type_used"]:
        result["rule_type_used"].insert(0, "NBC")
    if not result["rule_references"]:
        result["rule_type_used"] = []

    result["review_required"] = normalize_bool(result["review_required"])
    if not result["rule_references"]:
        result["review_required"] = True
    if result["review_required"]:
        result["review_reason"] = (
            None
            if result["review_reason"] is None
            else normalize_text(result["review_reason"])
        )
        if result["review_reason"] in {None, "Not specified"}:
            result["review_reason"] = "No retrieved NBC rule explicitly covered this room."
    else:
        result["review_reason"] = None

    result["assumptions"] = [
        str(item).strip()
        for item in normalize_list(result["assumptions"])
        if str(item).strip()
    ]
    result["confidence_reason"] = normalize_text(result["confidence_reason"], default="")
    if not result["confidence_reason"] and result["rule_references"]:
        result["confidence_reason"] = (
            "Retrieved NBC rule chunks directly matched the room type or hazards."
        )
    if not result["confidence_reason"]:
        result["confidence_reason"] = "No explicit NBC rule was retrieved for this room."

    return {key: result[key] for key in FIRE_REQUIREMENT_KEYS}


def normalize_room_output(
    room: dict[str, Any],
    llm_payload: dict[str, Any],
    retrieved_documents: list[Any],
) -> dict[str, Any]:
    """Create the exact top-level output schema for one room."""
    output = {
        "room_number": room.get("room_number"),
        "room_type": room.get("room_type"),
        "area_sqft": room.get("area_sqft"),
        "capacity": room.get("capacity"),
        "ceiling_height_ft": room.get("ceiling_height_ft"),
        "zone_type": room.get("zone_type"),
        "hazards": room.get("hazards") if isinstance(room.get("hazards"), list) else [],
        "geometry": room.get("geometry"),
        "code_agent_version": CODE_AGENT_VERSION,
        "fire_requirements": normalize_fire_requirements(
            llm_payload,
            retrieved_documents,
        ),
    }
    return {key: output[key] for key in ROOM_OUTPUT_KEYS}


def validate_room_output_schema(output: dict[str, Any]) -> None:
    """Validate that one room output preserves the Design Agent contract."""
    top_level_keys = list(output.keys())
    if top_level_keys != ROOM_OUTPUT_KEYS:
        raise RAGConfigurationError(
            "Room output schema mismatch. "
            f"Expected keys {ROOM_OUTPUT_KEYS}, got {top_level_keys}."
        )

    fire_requirements = output.get("fire_requirements")
    if not isinstance(fire_requirements, dict):
        raise RAGConfigurationError("fire_requirements must be a JSON object.")

    requirement_keys = list(fire_requirements.keys())
    if requirement_keys != FIRE_REQUIREMENT_KEYS:
        raise RAGConfigurationError(
            "fire_requirements schema mismatch. "
            f"Expected keys {FIRE_REQUIREMENT_KEYS}, got {requirement_keys}."
        )

    if not isinstance(output["hazards"], list):
        raise RAGConfigurationError("hazards must be a list.")
    if not isinstance(fire_requirements["sprinkler_required"], bool):
        raise RAGConfigurationError("sprinkler_required must be a boolean.")
    if not isinstance(fire_requirements["detector_required"], bool):
        raise RAGConfigurationError("detector_required must be a boolean.")
    if not isinstance(fire_requirements["manual_call_point_required"], bool):
        raise RAGConfigurationError("manual_call_point_required must be a boolean.")
    if not isinstance(fire_requirements["extinguisher_required"], bool):
        raise RAGConfigurationError("extinguisher_required must be a boolean.")
    if not isinstance(fire_requirements["special_suppression_required"], bool):
        raise RAGConfigurationError("special_suppression_required must be a boolean.")
    if not isinstance(fire_requirements["rule_type_used"], list):
        raise RAGConfigurationError("rule_type_used must be a list.")
    if not isinstance(fire_requirements["assumptions"], list):
        raise RAGConfigurationError("assumptions must be a list.")
    if not isinstance(fire_requirements["rule_references"], list):
        raise RAGConfigurationError("rule_references must be a list.")


def processing_error_output(room: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Return a valid room output when one room fails during processing."""
    payload = {
        "fire_requirements": {
            **default_fire_requirements(),
            "classification_confidence": "LOW",
            "confidence_reason": f"Room processing failed: {error}",
            "review_required": True,
            "review_reason": "Processing error",
        }
    }
    output = normalize_room_output(room, payload, retrieved_documents=[])
    validate_room_output_schema(output)
    return output


def compact_room_query(room: dict[str, Any]) -> str:
    """Convert room data into a retrieval query."""
    hazards = room.get("hazards") if isinstance(room.get("hazards"), list) else []
    hazard_text = ", ".join(str(hazard) for hazard in hazards) or "no listed hazards"
    parts = [
        f"room type {room.get('room_type', 'unknown')}",
        f"zone {room.get('zone_type', 'unknown')}",
        f"hazards {hazard_text}",
        "sprinkler detector portable extinguisher NBC Table 7",
    ]
    return " | ".join(parts)


def document_matches_room(document: Any, room: dict[str, Any]) -> bool:
    """Keep only retrieved chunks that explicitly mention this room category."""
    content = document.page_content.lower()
    candidates: list[str] = []

    room_type = room.get("room_type")
    if room_type:
        candidates.append(str(room_type).replace("_", " ").lower())

    hazards = room.get("hazards") if isinstance(room.get("hazards"), list) else []
    candidates.extend(str(hazard).replace("_", " ").lower() for hazard in hazards)

    normalized_candidates = [
        candidate
        for candidate in candidates
        if candidate and candidate not in {"unknown", "none", "no listed hazards"}
    ]
    return any(candidate in content for candidate in normalized_candidates)


def format_retrieved_context(documents: list[Any]) -> str:
    """Serialize rule chunks for the Gemini prompt."""
    if not documents:
        return "No retrieved NBC sections explicitly match this room."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        blocks.append(
            f"[Retrieved NBC Section {index}]\n"
            f"Source: {metadata.get('source', 'unknown')}\n"
            f"Rule ID: {metadata.get('rule_id', 'unknown')}\n"
            f"Reference: {metadata.get('reference', 'Not specified')}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(blocks)


@dataclass
class FireSafetyCodeRAG:
    """RAG-backed Code Agent for room fire requirements."""

    vectorstore: Any
    llm: Any
    k: int = 6

    def __post_init__(self) -> None:
        _, ChatPromptTemplate, _, _ = get_langchain_dependencies()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )

    def retrieve_rule_documents(self, room: dict[str, Any]) -> list[Any]:
        """Retrieve NBC rule chunks for one room."""
        query = compact_room_query(room)
        documents = self.vectorstore.similarity_search(query, k=self.k)

        unique_documents: list[Any] = []
        seen: set[str] = set()
        for document in documents:
            rule_id = str(document.metadata.get("rule_id") or document.page_content)
            if rule_id in seen:
                continue
            seen.add(rule_id)
            unique_documents.append(document)

        return [
            document
            for document in unique_documents
            if document_matches_room(document, room)
        ]

    def analyze_room(self, room: dict[str, Any]) -> dict[str, Any]:
        """Generate the strict fire-requirement JSON for one room."""
        documents = self.retrieve_rule_documents(room)
        messages = self.prompt.invoke(
            {
                "room_json": json.dumps(room, indent=2),
                "retrieved_context": format_retrieved_context(documents),
                "schema_instructions": SCHEMA_INSTRUCTIONS,
            }
        )
        response = self.llm.invoke(messages)
        content = message_content_to_text(getattr(response, "content", ""))
        llm_payload = extract_json_object(content)
        output = normalize_room_output(room, llm_payload, documents)
        validate_room_output_schema(output)
        return output

    def analyze_rooms(self, rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze all rooms independently, preserving schema on failures."""
        results: list[dict[str, Any]] = []
        for room in rooms:
            try:
                result = self.analyze_room(room)
            except Exception as error:
                result = processing_error_output(room, error)
            results.append(result)
        return results


def build_code_rag(
    vectorstore_dir: Path = CHROMA_DIR,
    model: str | None = None,
    temperature: float = 0,
    k: int = 6,
) -> FireSafetyCodeRAG:
    """Build the Agent 2 RAG chain."""
    ensure_gemini_api_key()
    vectorstore = load_vectorstore(vectorstore_dir)
    _, _, ChatGoogleGenerativeAI, _ = get_langchain_dependencies()
    llm = ChatGoogleGenerativeAI(
        model=model or get_chat_model(require_env_file=True),
        temperature=temperature,
    )
    return FireSafetyCodeRAG(vectorstore=vectorstore, llm=llm, k=k)


if __name__ == "__main__":
    try:
        from code_agent import load_rooms_from_file

        agent = build_code_rag()
        result = agent.analyze_rooms(load_rooms_from_file())
    except (ProjectConfigurationError, RAGConfigurationError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(json.dumps(result, indent=2))
