# W8D2 --- RAG Evaluation with Ragas

## Overview

This project evaluates a Retrieval-Augmented Generation (RAG) pipeline
using the **Ragas** framework. It compares a baseline retrieval
configuration with an optimised configuration and records evaluation
results for analysis.

The pipeline uses LangChain, ChromaDB, and a local Ollama model.
Evaluation questions and reference answers are stored in
`qa_pairs.json`.

## Evaluation metrics

  -----------------------------------------------------------------------
  Metric                  What it measures        Better result
  ----------------------- ----------------------- -----------------------
  Faithfulness            Whether the answer is   Higher
                          supported by the        
                          retrieved context       

  Answer relevancy        Whether the answer      Higher
                          addresses the question  

  Context precision       Whether retrieved       Higher
                          context is relevant and 
                          ranked usefully         

  Context recall          How much of the         Higher
                          reference information   
                          is retrieved            
  -----------------------------------------------------------------------

Scores are between `0` and `1`. A metric may be unavailable (`n/a`) when
Ragas cannot calculate a valid score for the evaluated samples.

## Requirements

-   Python and a virtual environment
-   Dependencies listed in `requirements.txt`
-   Ollama running locally
-   Ollama models: `llama3.2` and `nomic-embed-text`
-   Source PDFs placed in `pdfs/` before converting them to text

Install dependencies:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the models and start Ollama if it is not already running:

``` powershell
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

If Ollama is already running in the background, do not start a second
server.

## Prepare the source documents

1.  Place the project's source PDF documents in `pdfs/`.
2.  Convert the PDFs to text using the supplied script:

``` powershell
python pdf_to_txt.py
```

Ensure the expected text files are available under `data/` before
running evaluation.

## Run evaluation

Run the helper tests first:

``` powershell
pytest -q
```

Run a small smoke test:

``` powershell
python evaluate_ragas.py --label smoke --limit 2
```

Run the baseline configuration:

``` powershell
python evaluate_ragas.py --label baseline --chunk 1000 --k 3
```

Run the optimised configuration:

``` powershell
python evaluate_ragas.py --label optimised --chunk 1000 --k 5
```

Compare the results:

``` powershell
python compare.py baseline optimised
```

Here, `--chunk 1000` sets the chunk size to 1,000 characters and `k`
sets the number of retrieved chunks. The comparison uses the generated
summary JSON files in `results/`.

## Results

The completed run produced the following aggregate scores:

  Metric                Baseline (`k=3`)   Optimised (`k=5`)                       Change
  ------------------- ------------------ ------------------- ----------------------------
  Faithfulness                     0.646               1.000                       +0.354
  Answer relevancy                   n/a               0.817   Available in optimised run
  Context precision                1.000               1.000                       +0.000
  Context recall                   0.667               0.462                       -0.205

**Interpretation:** The optimised run improved the reported faithfulness
score and produced an answer-relevancy score. Context precision stayed
at 1.000, while context recall decreased from 0.667 to 0.462. Therefore,
increasing `k` from 3 to 5 did **not** improve every metric; the results
show a trade-off. Answer relevancy was unavailable for the baseline, so
a direct comparison for that metric cannot be made.

Some individual metric evaluations timed out or returned invalid values.
The summary reports the valid-sample count for each metric, so the
aggregate scores should be interpreted with that limitation in mind. Do
not treat missing values as zero.

## Output files

The evaluation creates result files under `results/`, including:

-   `ragas_baseline.csv` --- per-question baseline scores
-   `summary_baseline.json` --- baseline aggregate metrics
-   `ragas_optimised.csv` --- per-question optimised scores
-   `summary_optimised.json` --- optimised aggregate metrics
-   Smoke-test output for the small test run

Exact output filenames follow the labels passed to `evaluate_ragas.py`.

## Troubleshooting

-   **Slow evaluation / timeouts:** Local model inference and Ragas
    scoring can be slow on CPU. Keep Ollama running and start with the
    smoke test.
-   **Ollama connection errors:** Check that Ollama is running and that
    both required models are installed.
-   **Missing or weak context:** Confirm the PDFs were added and
    converted, and that the text files are in the expected `data/`
    location.
-   **Missing metric values:** Review the terminal output and
    per-question CSV; Ragas may fail to score some samples.
-   **Output parsing errors:** Inspect the generated answer and model
    output; malformed output can cause parsing failures.

## Suggested Git workflow

Use your assigned internship branch and inspect the changes before
committing:

``` powershell
git status
git add README.md
git commit -m "docs: document W8D2 Ragas evaluation"
git push -u origin feat/aiml-W8-siriyala
```

If the branch already tracks the remote, use `git push` instead of
`git push -u origin ...`. Add and commit the code, tests, and evaluation
artifacts separately as required by your internship instructions.

## Conclusion

The Ragas evaluation compares answer quality and retrieval quality
across two configurations. The observed results indicate stronger
faithfulness in the optimised run, unchanged context precision, and
lower context recall. Further tuning should focus on improving recall
while preserving faithfulness, and rerunning evaluation to obtain more
valid metric samples.
