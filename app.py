"""Streamlit UI for Agent 2 Fire Safety Layout Code Generation."""

from __future__ import annotations

import streamlit as st

from build_vectorstore import (
    VectorStoreBuildError,
    build_vectorstore,
    load_vectorstore_metadata,
    vectorstore_ready,
)
from code_agent import CodeAgentInputError, analyze_rooms, load_rooms_from_text
from rag_chain import RAGConfigurationError
from settings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ENV_PATH,
    INPUT_ROOMS_PATH,
    NBC_RULES_PATH,
    ProjectConfigurationError,
)


def load_default_room_json() -> str:
    """Load the sample room JSON shown in the input text area."""
    if INPUT_ROOMS_PATH.exists():
        return INPUT_ROOMS_PATH.read_text(encoding="utf-8")
    return "[]"


def render_sidebar() -> None:
    """Render local configuration and vector-store status."""
    st.sidebar.header("Agent 2")
    metadata = load_vectorstore_metadata()

    if st.sidebar.button("Rebuild NBC Vector Store", type="secondary"):
        try:
            with st.spinner("Embedding small NBC rule base..."):
                build_vectorstore()
            st.sidebar.success("NBC vector store rebuilt.")
        except (ProjectConfigurationError, VectorStoreBuildError, FileNotFoundError) as error:
            st.sidebar.error(str(error))

    st.sidebar.divider()
    st.sidebar.metric("NBC Rule Chunks", metadata.get("rule_count", 0))
    st.sidebar.caption(f"Rules file: {NBC_RULES_PATH.name}")
    st.sidebar.caption(
        f"Embeddings: {metadata.get('embedding_model') or DEFAULT_EMBEDDING_MODEL}"
    )
    st.sidebar.caption(f"Chat: {metadata.get('chat_model') or DEFAULT_CHAT_MODEL}")

    st.sidebar.divider()
    if ENV_PATH.exists():
        st.sidebar.success(".env found")
    else:
        st.sidebar.warning(".env missing")

    if NBC_RULES_PATH.exists():
        st.sidebar.success("nbc_rules.txt found")
    else:
        st.sidebar.warning("nbc_rules.txt missing")

    if vectorstore_ready():
        st.sidebar.success("Chroma vector store ready")
    else:
        st.sidebar.warning("Chroma vector store missing")


def summary_rows(result: list[dict]) -> list[dict]:
    """Build the summary table rows requested by the UI spec."""
    rows: list[dict] = []
    for room in result:
        requirements = room["fire_requirements"]
        rows.append(
            {
                "Room": room["room_number"],
                "Room Type": room["room_type"],
                "Mapped Category": requirements["mapped_category"],
                "Sprinkler": requirements["sprinkler_required"],
                "Detector": requirements["detector_required"],
                "Extinguisher": requirements["extinguisher_required"],
                "Confidence": requirements["classification_confidence"],
                "NBC References": "; ".join(requirements["rule_references"]),
            }
        )
    return rows


def output_metrics(result: list[dict]) -> dict[str, int]:
    """Calculate batch metrics for the Streamlit result view."""
    high_confidence = 0
    review_required = 0

    for room in result:
        requirements = room["fire_requirements"]
        if requirements["classification_confidence"] == "HIGH":
            high_confidence += 1
        if requirements["review_required"]:
            review_required += 1

    return {
        "total_rooms": len(result),
        "high_confidence": high_confidence,
        "review_required": review_required,
    }


def render_results(result: list[dict]) -> None:
    """Render summary and pretty JSON output."""
    metrics = output_metrics(result)
    total_col, high_col, review_col = st.columns(3)
    total_col.metric("Total rooms processed", metrics["total_rooms"])
    high_col.metric("HIGH confidence rooms", metrics["high_confidence"])
    review_col.metric("Rooms requiring review", metrics["review_required"])

    st.subheader("Summary")
    st.dataframe(summary_rows(result), width="stretch", hide_index=True)

    st.subheader("Agent 2 JSON Output")
    st.json(result)

    with st.expander("NBC References"):
        for room in result:
            requirements = room["fire_requirements"]
            references = requirements["rule_references"] or ["No NBC references used"]
            st.markdown(f"**Room {room['room_number']}**")
            for reference in references:
                st.markdown(f"- {reference}")


def render_main() -> None:
    """Render the main Streamlit experience."""
    st.title("Fire Safety Code Agent")
    st.caption(
        "Agent 2 only: Agent 1 room JSON array -> RAG over small NBC rules -> strict JSON"
    )

    room_json = st.text_area(
        "Room JSON array",
        value=load_default_room_json(),
        height=360,
        placeholder='[{"room_number": "126A", "room_type": "Mechanical"}]',
    )

    if st.button("Analyze Fire Requirements", type="primary"):
        try:
            rooms = load_rooms_from_text(room_json)
            with st.spinner("Retrieving NBC rules and calling Gemini 2.5 Flash..."):
                result = analyze_rooms(rooms)
        except (
            CodeAgentInputError,
            FileNotFoundError,
            ProjectConfigurationError,
            RAGConfigurationError,
            VectorStoreBuildError,
            ValueError,
        ) as error:
            st.error(str(error))
            return

        render_results(result)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Fire Safety Code Agent", layout="wide")
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
