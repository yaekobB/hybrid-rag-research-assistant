# LLM-Powered Research Assistant

A modular, RAG-based **research assistant** that combines:

- Local **Retrieval-Augmented Generation (RAG)** over user-uploaded PDFs  
- **Web RAG** using Tavily for fresh external knowledge  
- A simple **task-type planner** (QA, summary, compare, plan, survey)  
- A structured **Writer** that produces academic-style answers  
- A **Reflection** module that critiques and improves those answers  
- A Streamlit **web UI** with LLM parameter control and PDF export

This project was developed to demonstrate
multi-step reasoning, contextual grounding, and transparent, research-style outputs.

---

## 1. Features

- 🔍 **Local RAG over PDFs**
  - Upload course papers, survey articles, or project documents.
  - PDFs are chunked, embedded (`all-MiniLM-L6-v2`), and stored in an in-memory vector store.
  - Semantic search retrieves the most relevant chunks as context.

- 🌐 **Web RAG (Tavily API)**
  - Uses Tavily to fetch up-to-date web evidence for queries that go beyond local PDFs.
  - Web results are turned into numbered context snippets with URLs for citation.

- 🧠 **Task-Type Planner**
  - Simple rule-based classifier that maps each query to a task type:
    - `qa` – academic-style explanation
    - `summary` – concise summary
    - `compare` – comparison of methods/approaches
    - `plan` – high-level research/project plan
    - `survey` – mini literature-style overview

- ✍️ **Writer Module (Ollama Llama 3.2)**
  - Generates structured answers using task-specific templates:
    - 5-section QA (Introduction, Key Concepts, Evidence, Limitations, Conclusion)
    - Summary, comparison, research plan, or survey structures.
  - Uses inline numeric citations like `[1]`, `[2]` tied to retrieved evidence.

- 🔁 **Reflection Module**
  - A second LLM pass that:
    - Critiques the draft answer (clarity, grounding, hallucinations).
    - Produces an **Improved Answer** while keeping the same structure.
  - The **final answer** is the improved one; draft and critique are visible in debug mode.

- 🖥️ **Streamlit UI**
  - Text box for research questions.
  - PDF upload + ingestion controls.
  - Source selection: `local`, `web`, `hybrid`.
  - **LLM parameters in the UI**:
    - `temperature` (slider)
    - `max_tokens` (numeric input)
  - Answer panel + Reference list.
  - Debug panel (optional) with:
    - Task type
    - Draft answer
    - Reflection critique
    - Raw retrieval results
  - “Download as PDF” button (Question + Answer + References).

---

## 2. Project Structure

Below is the **typical folder structure** of the project (you can adapt names if needed):

```text
project-root/
├─ app/
│  ├─ __init__.py
│  ├─ config.py                # Paths, model name, Tavily key, etc.
│  ├─ logging_config.py        # Central logging setup
│  │
│  ├─ pipeline/
│  │  ├─ __init__.py
│  │  └─ orchestrator.py       # Main pipeline: planning, retrieval, writer, reflection
│  │
│  ├─ tools/
│  │  ├─ __init__.py
│  │  ├─ local_rag_langchain.py  # LangChain PDF ingestion + semantic search
│  │  ├─ rag_web.py              # Tavily-based web retrieval
│  │  └─ export_utils.py         # Helper for exporting answers to PDF
│  │
│  └─ ui/
│     ├─ __init__.py
│     └─ ui_streamlit.py       # Streamlit front-end (tabs, controls, debug)
│
├─ data/
│  ├─ pdfs_uploaded/           # Uploaded PDFs (input)
│  └─ vectorstore/             # (Optional) persistent embeddings/index
│
├─ logs/                       # Application logs (optional)
│
├─ requirements.txt            # Python dependencies
├─ README.md                   # Project documentation
└─ .env.example                # Example for environment variables (Tavily API key, etc.)
```


---

## 3. Installation

1. **Clone the repository**

```bash
git clone https://github.com/yaekobB/hybrid-rag-research-assistant.git
cd hybrid-rag-research-assistant
```

2. **Create and activate a virtual environment** (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install and pull the LLM with Ollama**

- Install Ollama from: https://ollama.com  
- Pull the model used in this project (for example):
```bash
ollama pull llama3.2
```

5. **Set Tavily API key (if using Web RAG)**

Create a `.env` file or set an environment variable:

```bash
export TAVILY_API_KEY="your_tavily_api_key_here"
```

On Windows (PowerShell):

```powershell
$env:TAVILY_API_KEY="your_tavily_api_key_here"
```

Make sure `config.py` reads this key correctly.

---

## 4. How to Run

1. Start the **Ollama** server (if not already running):
```bash
ollama serve
```

2. From the project root, launch the **Streamlit app**:

```bash
python -m streamlit run app/ui/ui_streamlit.py
```

3. Open the URL shown in the terminal (e.g. `http://localhost:8501`).

---

## 5. Usage Guide

1. **Upload PDFs (optional but recommended)**
   - Use the sidebar “Upload PDFs” area.
   - Click *“Ingest PDFs into Local RAG (LangChain)”* to index them.

2. **Choose source mode**
   - `local`  – only your uploaded PDFs.
   - `web`    – only Tavily web search.
   - `hybrid` – combine both.

3. **Tune LLM parameters in the UI**
   - **Temperature**: lower = more deterministic, higher = more creative.
   - **Max tokens**: upper bound on answer length.

4. **Ask your research question**
   - Examples:
     - *“What are recent research directions in AI for medical imaging?”*
     - *“Summarize this paper in 5 bullet points.”*
     - *“Compare RAG and Self-RAG for factuality.”*
     - *“Give me a research plan for a thesis on chest X-ray report generation.”*

5. **Inspect the output**
   - Main panel: final, **refined** answer (after reflection).
   - References: numbered evidence sources (PDFs and/or web URLs).
   - Debug mode (checkbox in sidebar):
     - Shows *task type*, *draft answer*, *reflection critique*, and *raw retrieval*.

6. **Export as PDF**
   - Use the *“Download Answer as PDF”* button to save a clean report containing:
     - Question
     - Final answer
     - References

---

## 6. Debugging & Troubleshooting

- **No answer / error during retrieval**
  - Check that PDFs were ingested (for `local` mode).
  - Check your Tavily API key (for `web` or `hybrid` mode).
  - Look at the terminal logs for stack traces.

- **Ollama connection issues**
  - Ensure `ollama serve` is running.
  - Confirm `LLM_MODEL` in `config.py` matches a pulled model (e.g. `llama3.2`).

- **Strange or generic answers**
  - Enable **debug mode** to see:
    - What documents/web pages were retrieved.
    - The draft answer before reflection.
    - The critique, which may reveal missing context or misaligned queries.

---

## 7. Future Improvements

- Learnable task-type classifier (instead of rule-based planner).
- More advanced retrieval (multi-query, reranking, keyword + dense hybrid).
- Richer UI (tabbed chat history, document browser, citation filtering).
- Logging and evaluation scripts for quantitative assessment of answer quality.

---
