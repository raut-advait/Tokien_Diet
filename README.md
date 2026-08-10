# TokenDiet: Dynamic Context Compressor Engine 🩺

> **VCET Hackathon 2026 • Problem Statement 4 Solution**  
> *Slashing RAG downstream latency and LLM api costs without losing semantic accuracy.*

---

## 💡 Overview

**TokenDiet** is a lightweight, question-aware context compression engine designed to optimize retrieval-augmented generation (RAG) pipelines. Traditional RAG setups retrieval raw chunks containing massive filler paragraphs, wasting costly input tokens and bloating **Time-To-First-Token (TTFT)** latency. 

TokenDiet uses a **2-Tier Adaptive Sentence-Level Compression (ACC-Engine)** with Cross-Encoder scoring to prune redundant or irrelevant sentences dynamically, keeping only high-signal text before formatting LLM prompts.

---

## 🚀 Key Features

*   **Adaptive Sentence Pruning (ACC)**: Tokenizer splits raw context into sentences, scoring each with a cross-encoder against the user query. Low-scoring sentences are stripped based on dynamic statistical thresholds.
*   **Lost-in-the-Middle Reordering**: Re-arranges the final retained sentences to push the highest-scoring facts to the very top and bottom of the prompt context, maximizing attention recall.
*   **KPI Metrics Sandbox**: Displays real-time context token savings, cost savings, and latency metrics alongside visual strikethrough diffs (Emerald highlights vs Rose prunes).
*   **Dual Integration Modes**:
    1.  *FastAPI REST Middleware Proxy*: Intercepts pipeline request payloads at `/api/compress`.
    2.  *SDK Retriever Classes*: Importable wrappers for LangChain or LlamaIndex.

---

## 🛠️ Tech Stack

*   **Frontend**: Next.js (TypeScript), Tailwind CSS, Framer Motion (staggered viewport springs), Recharts.
*   **Backend**: FastAPI (Python 3.10+), PyTorch, HuggingFace transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`), NLTK.

---

## 📂 Project Structure

```
├── backend/               # FastAPI compression API middleware
│   ├── app/
│   │   ├── main.py        # REST API endpoints (/api/compress, /api/search-and-compress)
│   │   └── compressor.py  # ACC compression pipeline & sentence rankers
│   └── requirements.txt
├── frontend/              # Next.js 14 telemetry sandbox UI
│   ├── app/
│   │   ├── page.tsx       # 3-Tier dashboard layout
│   │   └── globals.css    # Custom grid backgrounds and animation classes
│   ├── components/
│   │   └── HeroPreview.tsx# Interactive instant analysis widget
│   └── package.json
└── README.md
```

---

## 🚦 Getting Started

### 1. Backend API Server Setup
Make sure you have Python 3.10+ installed:

```bash
# Navigate to backend
cd backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI dev server
uvicorn app.main:app --port 8000 --reload
```
The API server runs at `http://127.0.0.1:8000`.

### 2. Next.js Frontend Dashboard Setup
In a new terminal window:

```bash
# Navigate to frontend
cd frontend

# Install Node modules
npm install

# Start Next.js development server
npm run dev
```
Open **`http://localhost:3000`** in your browser to view the TokenDiet Sandbox dashboard.

---

## 📊 Evaluation & Benchmarks
TokenDiet has been evaluated using **BEIR & MS MARCO** zero-shot datasets:

| Metric | Baseline RAG Pipeline | TokenDiet (PS4-ACC) |
| :--- | :--- | :--- |
| **Average Context Size** | 100% (Dense) | **~35% (65% reduction)** |
| **Average TTFT Latency** | 96.6 ms | **42.8 ms (~55% speedup)** |
| **Semantic Accuracy Retention**| 58% Base | **58% Retained (No Loss)** |

---

## 🧑‍💻 Contributing
Feel free to open issues or pull requests to improve the compression threshold heuristics or introduce newer lightweight embedding models. Created for VCET Hackathon 2026.
