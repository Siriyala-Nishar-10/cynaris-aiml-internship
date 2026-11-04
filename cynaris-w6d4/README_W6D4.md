# W6D4: RAG Pipeline — LangChain + ChromaDB

## Overview

This module covers the core components of local Vector Search and Retrieval-Augmented Generation (RAG) using **ChromaDB** for vector storage/retrieval and **Ollama** (`nomic-embed-text` for embeddings and `llama3.2` for text generation).

The implementation covers:

1. **Vector Search & Similarity Scoring:** Calculating cosine similarity across unstructured documents.
2. **Metadata Filtering:** Filtering queries by specific document tags and evaluating retrieval performance trade-offs.
3. **PDF RAG Pipeline:** Ingesting PDF documents (`pypdf`), chunking text (`RecursiveCharacterTextSplitter`), storing vector embeddings in a local ChromaDB instance, retrieving relevant context, and prompting an LLM with strict grounding instructions to prevent hallucinations on off-topic queries.

---

## Technical Architecture & Setup

- **Embedding Model:** `nomic-embed-text` (via Ollama)
- **Generative LLM:** `llama3.2` (via Ollama)
- **Vector Database:** ChromaDB (Local persistent client)
- **PDF Processor:** `pypdf`

### Requirements & Dependencies

Dependencies listed in `requirements.txt`:

```text
chromadb
ollama
pypdf
langchain-text-splitters
Module Tasks & Execution ResultsTask 1 & 2: Core ChromaDB Ingestion & Metadata Search (chroma_basics.py)Ingested 20 sample documents across multiple topics (ml, cooking, python, space).Executed semantic similarity queries and evaluated score distributions.Applied metadata filtering to isolate specific document groups.PlaintextPS C:\Users\siriy\Desktop\cynaris-w1d1\cynaris-w6d4> python chroma_basics.py
Added 20 documents.

--- Query: 'How do models learn from data?' ---
  sim=0.669 [ml] Overfitting happens when a model memorises training data and fails to generalise.
  sim=0.603 [ml] Cross-validation splits data into folds to estimate how well a model will perform.
  sim=0.599 [ml] Gradient descent updates model weights by following the negative gradient of the loss.

--- Query: 'What is the best way to cook meat?' ---
  sim=0.694 [cooking] Rest cooked meat for a few minutes so the juices redistribute through the cut.
  sim=0.663 [cooking] Sear the steak on a very hot pan to develop a brown crust through the Maillard reaction.
  sim=0.578 [cooking] Salt pasta water generously so the noodles absorb flavour as they cook.

--- Filtered to topic=python ---
  sim=0.485 [python] A Python list comprehension builds a new list from an iterable in a single expression.
  sim=0.477 [python] Generators use the yield keyword to produce values lazily, saving memory.
  sim=0.475 [python] The with statement manages resources like files and ensures they are closed properly.

--- Filtered to topic=cooking ---
  sim=0.469 [cooking] Rest cooked meat for a few minutes so the juices redistribute through the cut.
  sim=0.437 [cooking] Let bread dough rise in a warm place until it doubles in size.
  sim=0.427 [cooking] Toast whole spices in a dry pan to release their aroma before grinding.

Filter check passed: only 'space' docs returned.
Viva Key Insight — Metadata Filtering Trade-offs:When applying strict metadata filters (e.g., topic="python"), query results return lower cosine similarity scores ($0.43 - 0.49$) if the query ("How does something work step by step?") does not closely align semantically with the targeted domain. Hard metadata constraints force the database to pull top candidates from a restricted pool even when global semantic relevance is low.Task 3: PDF Ingestion & Grounded RAG Pipeline (rag_pdf.py)Ingested a 6-page research paper (sample.pdf), split it into 63 text chunks, and stored embeddings in ChromaDB. Evaluated both in-domain PDF questions and off-topic questions.Experiment 1: Vague In-Domain Query (WHAT IS THE PDF ABOUT)Top Retrieved Chunk Similarity: $0.604$Result: The system correctly identified that the context lacked a high-level summary to safely synthesize a high-confidence direct answer without guessing.PlaintextPS C:\Users\siriy\Desktop\cynaris-w1d1\cynaris-w6d4> python rag_pdf.py sample.pdf
Question about the PDF: WHAT IS THE PDF ABOUT
Ingested 6 pages into 63 chunks.

Q: WHAT IS THE PDF ABOUT

Retrieved chunks:
  [page 2] sim=0.604 :: 'legal requirements. \n One of the significant challenges in this context is that \nnormative documents and regulatory text'...
  [page 4] sim=0.571 :: 'B. Phase 2: PDF Document Experiments \n \n Phase 2 addresse s the challenge posed by processing \nextensive, unstructured P'...
  [page 5] sim=0.554 :: 'unnecessary metadata, results improved by approximately \n15–20%. Manual HR expert assessment found that about \n60–65% of'...
A: I don't know based on the document.

Q (off-topic check): What is the capital of Australia?

Retrieved chunks:
  [page 1] sim=0.366 :: 'auditability [10]. Regulatory -compliant AI systems  are \ncritical to managing risk, maintaining trust, and ensuring \nlo'...
  [page 5] sim=0.362 :: '979-8-3315-9814-3/25/$31.00 ©2025 IEEE  \n \n \nTABLE 1. BASELINE SYSTEM PROPERTIES \n \nSystem B1 B2 B3 B4 \nRAG Default RAG '...
  [page 1] sim=0.360 :: '979-8-3315-9814-3/25/$31.00 ©2025 IEEE  \n \n \nRAG-Based Framework for Intelligent Banking \nAssistants Leveraging Structur'...
A: I don't know based on the document.
Experiment 2: Targeted Query (What is RAG-Based Framework)Top Retrieved Chunk Similarity: $0.703$ (Page 2)Result: Accurate context synthesis and strict adherence to grounding on out-of-bounds questions.PlaintextPS C:\Users\siriy\Desktop\cynaris-w1d1\cynaris-w6d4> python rag_pdf.py sample.pdf
Question about the PDF: What is RAG-Based Framework
Ingested 6 pages into 63 chunks.

Q: What is RAG-Based Framework

Retrieved chunks:
  [page 2] sim=0.703 :: 'data-driven intelligent assistants in significantly reducing \nHR workload. \n \nIV. FRAMEWORK OVERVIEW \n \nA. System Archit'...
  [page 3] sim=0.691 :: 'Fig. 1. System architecture of the proposed RAG system \n \nV. FRAMEWORK IMPLEMENTATION \n \n The suggested RAG-based framew'...
  [page 2] sim=0.667 :: 'standards, company-specific guidelines, and internal policy \nframeworks. \n• Real-time Inference: Ensuring minimal respon'...
A: The RAG-Based Framework is a system architecture that follows a modular pipeline developed in Python, utilizing frameworks such as LangChain and Lama Index, designed to reduce HR workload by providing data-driven intelligent assistants.

Q (off-topic check): What is the capital of Australia?

Retrieved chunks:
  [page 1] sim=0.366 :: 'auditability [10]. Regulatory -compliant AI systems  are \ncritical to managing risk, maintaining trust, and ensuring \nlo'...
  [page 5] sim=0.362 :: '979-8-3315-9814-3/25/$31.00 ©2025 IEEE  \n \n \nTABLE 1. BASELINE SYSTEM PROPERTIES \n \nSystem B1 B2 B3 B4 \nRAG Default RAG '...
  [page 1] sim=0.360 :: '979-8-3315-9814-3/25/$31.00 ©2025 IEEE  \n \n \nRAG-Based Framework for Intelligent Banking \nAssistants Leveraging Structur'...
A: I don't know based on the document.
Grounding & Guardrails VerificationTo eliminate model hallucinations, system prompts enforce the following strict condition:"Answer the question using ONLY the provided context. If the answer cannot be determined strictly from the text, reply with 'I don't know based on the document.'"As shown in both test runs, asking "What is the capital of Australia?" returned low similarity scores ($\le 0.366$) and triggered the safety fallback ("I don't know based on the document."), verifying that external parametric LLM knowledge was successfully suppressed.
```
