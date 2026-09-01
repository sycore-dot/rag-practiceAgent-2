# Fire Safety Layout Generation - Agent 2

This project implements only Agent 2, the Code Agent. It assumes Agent 1 has
already produced room JSON from a floor plan. Agent 3 / Design Agent is not
implemented here.

## Architecture

```text
Input JSON
-> Convert room information into retrieval queries
-> Retrieve relevant chunks from small nbc_rules.txt knowledge base
-> Gemini 2.5 Flash
-> Strict JSON fire requirements
```

The knowledge base is intentionally small and local. The agent must use only
retrieved NBC rule chunks and return valid JSON for downstream layout generation.

## Files

```text
input_rooms.json        Sample Agent 1 room JSON
nbc_rules.txt           Small NBC rule knowledge base
build_vectorstore.py    Embeds nbc_rules.txt into ChromaDB
rag_chain.py            RAG retrieval, Gemini prompt, schema normalization
code_agent.py           CLI and Python API for Agent 2
app.py                  Streamlit UI
settings.py             Paths, model names, and .env loading
.env.example            Environment variable template
requirements.txt        Python dependencies
README.md               This guide
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

Optional model overrides:

```text
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
```

## Build The NBC Vector Store

```powershell
python build_vectorstore.py
```

This creates a local `chroma_db/` directory from `nbc_rules.txt`.

## Run The Code Agent

```powershell
python code_agent.py --input input_rooms.json
```

Write output to a file:

```powershell
python code_agent.py --input input_rooms.json --output fire_requirements.json
```

The CLI auto-builds Chroma if it is missing. Use `--no-auto-build` to fail
instead.

## Run Streamlit

```powershell
streamlit run app.py
```

Paste Agent 1 room JSON and click **Analyze Fire Requirements**.

The UI shows:

- Mapped category
- Sprinkler requirement
- Detector requirement
- Extinguisher requirement
- Confidence
- NBC references
- Pretty JSON output

## Output Contract

Every room is returned with the original Agent 1 fields plus:

```json
{
  "code_agent_version": "rag_v2",
  "fire_requirements": {
    "mapped_category": "MECHANICAL_ROOM",
    "hazard_class": "Not specified",
    "sprinkler_required": true,
    "sprinkler_system_type": "Wet pipe",
    "detector_required": true,
    "detector_type": "Heat detector preferred",
    "manual_call_point_required": false,
    "extinguisher_required": true,
    "special_suppression_required": false,
    "special_suppression_type": null,
    "classification_confidence": "HIGH",
    "confidence_reason": "Retrieved NBC rule chunks directly matched the room type or hazards.",
    "rule_type_used": [
      "NBC"
    ],
    "review_required": false,
    "review_reason": null,
    "assumptions": [],
    "rule_references": [
      "NBC 2016 Part 4 Clause 5.1.3 Table 7",
      "NBC 2016 Part 4 Clause 5.1.1(a) Table 7"
    ]
  }
}
```

If no retrieved NBC rule explicitly applies, the agent marks the room for review
instead of inventing requirements.
