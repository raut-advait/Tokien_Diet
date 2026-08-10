# Product Context: Token-Diet Dynamic Context Compressor (PS4)

## Core Value Proposition
RAG applications suffer from token bloat, high Time-To-First-Token (TTFT) latency, high LLM API costs, and performance degradation when relevant information is buried in the middle of long contexts ("lost-in-the-middle").
The **Token-Diet Dynamic Context Compressor** performs **post-retrieval, sentence-level semantic pruning** using a hybrid scoring system (Cross-Encoder + BM25) to strip out filler and low-relevance sentences before forwarding the prompt to the LLM, reducing context length by up to 70% while maintaining query-relevance and retrieval quality.

## Features
- **Sentence-Level Splitting & Scoring**: Parses retrieved documents into sentences and scores each sentence against the user query.
- **Hybrid Scorer**: Fuses dense semantic scores from a Cross-Encoder with lexical matches from BM25.
- **Dynamic Compression Window**: Adaptive pruning that dynamically scales based on a target compression ratio or budget constraint.
- **Visual Diff Dashboard**: Highlights retained vs. pruned content to build operator trust.
- **QA Benchmark Suite**: Automatically profiles token savings, TTFT improvements, and cost reduction.
