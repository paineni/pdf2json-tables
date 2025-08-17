# 📄 PDF Table → 🧩 Structured JSON (LangGraph + docTR + LLM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OCR](https://img.shields.io/badge/OCR-docTR%20(Transformers)-orange)
![Graphs](https://img.shields.io/badge/Workflow-LangGraph-9cf)
![LLM](https://img.shields.io/badge/LLM-Gemini%20(OpenAI%20compat)-purple)
![Pydantic](https://img.shields.io/badge/Schema-Pydantic-informational)
![JSON](https://img.shields.io/badge/Output-Structured%20JSON-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Open Source Love](https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F-Open%20Source-pink)

**Open-source pipeline to extract *structured JSON* from complex PDF tables.**  
It runs OCR with **docTR**, flattens the spatial layout, then uses an **LLM with a strict Pydantic schema** to reconstruct full cells with row/column headers. Built as a **LangGraph** workflow with validation & retry.

## ✨ What it does

- **Reads complex PDF tables** (multi-level headers, merged cells, irregular layouts).
- **Runs OCR** via **docTR** and exports a rich layout (words + coordinates).
- **Flattens layout** to a clean list of `{text, x, y, width, height, conf}` for prompting.
- **LLM reconstruction**: prompts a Gemini-compatible endpoint and enforces a **Pydantic schema** for `TableStructure -> cells[value, row_headers, col_headers]`.
- **Validation + retry**: checks for **missing OCR texts** and performs a single corrective pass before persisting JSON.
- **Outputs**:
  - `outputs/doctr_output.json` – raw OCR export  
  - `outputs/structured_output.json` – *final* structured table

## 🧠 How it works (modules)

- `pdf2json.py` — orchestrates the **LangGraph**: nodes `ocr → flatten → llm → validate → persist`, with a conditional edge for retry. Also writes a **Mermaid graph PNG** of the flow.
- `utils.py` — defines **Pydantic** models `CellStructure`, `TableStructure` and the **layout flattener** from docTR’s nested blocks/lines/words.
- `system_message.txt` — the **LLM system prompt** instructing how to map spatial words to full cells and how to handle missing values on retry.

## 🗂 Project structure

```
.
├── pdf2json.py               # main pipeline (LangGraph + nodes)
├── utils.py                  # Pydantic schemas + docTR layout flattener
├── system_message.txt        # LLM system prompt (schema + retry guidance)
├── Table-Example-R.pdf       # sample PDF with complex table (input)
├── outputs/
│   ├── doctr_output.json     # raw OCR export
│   ├── structured_output.json# final structured table (output)
│   └── graph_flow.png        # auto-generated Mermaid diagram of the graph
└── README.md
```

> You can swap the sample PDF for your own. The pipeline reads the path from code (`initial_state["pdf_path"]`).

## 🔧 Installation

You can set up the environment in two ways:

### Option 1 — Using Conda (recommended)

```bash
# Create environment from environment.yaml
conda env create -f environment.yaml

# Activate the environment (as defined in environment.yaml)
conda activate myenv
```

### Option 2 — Using pip + requirements.txt

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install all dependencies
pip install -U pip
pip install -r requirements.txt
```

> **Note**: If you use a CUDA-specific PyTorch build, install it first, then run `pip install -r requirements.txt --no-deps`.


## 🔐 Environment

This repo uses a **Gemini-compatible OpenAI endpoint**. Set:

```bash
export GEMINI_API_KEY=your_api_key_here
```

- The code points `openai_api_base` to `https://generativelanguage.googleapis.com/v1beta/openai/` and model `gemini-2.5-pro`. Adjust as needed.

## 🚀 Usage

```bash
python pdf2json_table_exractor.py
```

What happens:

1) **OCR** on the input PDF → `outputs/doctr_output.json`  
2) **Flatten** to positional elements  
3) **LLM Structured Output** → `TableStructure(cells=...)`  
4) **Validate & Retry** if some OCR words were not mapped  
5) **Persist** → `outputs/structured_output.json` (final)

You’ll also get a generated **workflow diagram** at `outputs/graph_flow.png`.

## 🧪 Example I/O

- **Input PDF**: `Table-Example-R.pdf` (complex table)  
- **Output JSON**: `outputs/structured_output.json` — cells with `value`, `row_headers[]`, `col_headers[]`, suitable for analytics/ETL.

## 📦 Programmatic use (advanced)

Import and feed your own PDF path/state:

```python
from pdf2json import State, graph

state = State(pdf_path="my_table.pdf", retry_count=0, missing_texts=[])
final = graph.invoke(state)
print(final["llm_output"].model_dump())
```

## 🛠️ Why this approach?

- **OCR + Spatial Reasoning**: docTR provides high-quality word boxes; we preserve **positions** to recover headers & hierarchies.
- **Schema-first LLM**: Pydantic-bound output → ★ deterministic JSON structure.
- **Guardrails**: Missing-value validation and **one-shot retry** reduce silent losses common in table extraction.
- **Composable Graph**: Each step is a node; swap OCR/LLM backends or extend with post-processing easily.


## 🤝 Contributing

PRs and issues welcome!  
Ideas: new prompts, better validation, extra schemas (multi-table pages), performance tweaks, or alternative LLM providers.

## 📜 License

**MIT** — free to use, modify, and distribute.

### 💡 Quick tips

- If your PDFs are scans with **very small text**, try higher DPI rendering before OCR.  
- For very deep header hierarchies, consider a 2-pass prompt (first detect header bands, then fill cells).  
- Keep an eye on `missing_texts` in logs; it’s your signal for iterative improvements.
