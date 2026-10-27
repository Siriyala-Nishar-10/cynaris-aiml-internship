# W9D2: Research Crew - Without vs With Web Search

Generated: 2026-10-27 17:05  
Topic: Retrieval-Augmented Generation (RAG) best practices for small local LLMs  
Model: ollama/llama3.2:3b

> The metrics below are crude proxies. They cannot tell whether a URL is real or whether it
> supports the claim beside it. Complete the manual checks at the end.

## Metrics

| Metric | Without search | With search |
|---|---|---|
| Word count | 323 | 430 |
| URLs cited | 0 | 0 |
| Unique URLs | 0 | 0 |
| Unique domains | 0 | 0 |
| Has Sources section | True | True |
| States 'no sources consulted' | False | False |

## Output: Without web search

### Issues found

* The claim that RAG models often rely on a large-scale knowledge base to generate text is not supported by the research notes.
* The statement that RAG models can be optimized for small local LLMs by using a combination of pre-trained language models and fine-tuned models is not supported by the research notes.
* The claim that RAG models can be optimized for small local LLMs by using techniques like knowledge graph-based methods and graph neural networks is not supported by the research notes.

### Retrieval-Augmented Generation (RAG) Best Practices for Small Local LLMs
================================================================================

RAG models can benefit from using techniques like entity disambiguation and named entity recognition to improve accuracy and reduce errors.

**Knowledge Base Selection and Curation**
----------------------------------------

For small local LLMs, using a smaller, more focused knowledge base may be more effective than relying on a large-scale knowledge base. It's essential to carefully select the most relevant documents or knowledge sources for the RAG model to avoid over-reliance on biased or outdated information.

**Model Optimization and Fine-Tuning**
--------------------------------------

It's often more practical for small local LLMs to use a combination of pre-trained language models and fine-tuned models rather than relying on a single, large-scale model. This approach can help improve accuracy and reduce errors.

**Data Inclusion and Performance Monitoring**
--------------------------------------------

RAG models can be improved by incorporating additional data sources, such as user-generated content or expert opinions, to increase diversity and accuracy. When using RAG models for small local LLMs, it's crucial to monitor and evaluate performance regularly to ensure that the model is meeting its intended goals and not introducing unintended biases.

**Security and Integrity Measures**
---------------------------------

To ensure the security and integrity of RAG models, it's essential to implement robust data protection measures, such as encryption and access controls, to prevent unauthorized access or data breaches.

Sources:
- No specific sources were consulted, but the research notes provide general information on the topic.

## Output: With web search

## Issues found

* The claim that Qwen 3 14B or Gemma 3 12B are the best instruction followers for 12-16GB is repeated without clear evidence. The source [1] only mentions Command R 35B as the local model with built-in citation grounding for 24GB, but does not provide information on instruction followers for smaller models.
* The claim that Command R 35B returns inline source references instead of hallucinating them for 24GB is repeated without clear evidence. The source [1] only mentions this for 24GB, but does not provide information on smaller models.
* The claim that Ollama allows you to run everything privately on your machine without cloud costs is not supported by the research notes. The search request for Ollama and local RAG pipeline failed to find any relevant information.
* The claim that Pydantic AI is used for agentic RAG is not supported by the research notes. The search request for Pydantic AI and agentic RAG failed to find any relevant information.
* The claim that the project is an experimental sandbox for testing out ideas related to running local LLMs with Ollama and Pydantic AI is not supported by the research notes. The research notes do not provide any information on this project.

## Retrieval-Augmented Generation (RAG) Best Practices for Small Local LLMs

For small local LLMs, effective Retrieval-Augmented Generation (RAG) best practices are crucial for achieving optimal performance and efficiency.

## Instruction Followers and Model Parameters

When it comes to small local LLMs, the choice of instruction followers and model parameters is crucial. However, the research notes do not provide clear evidence on the best instruction followers for smaller models. Further research is needed to determine the optimal instruction followers for small local LLMs.

## Building a Local RAG Pipeline

To build a local RAG pipeline, consider using Ollama, but the research notes do not provide clear information on how to set up a local RAG pipeline using Ollama. Further research is needed to determine the best practices for setting up a local RAG pipeline.

## Best Practices and Resources

For more information on building document Q&A systems, vector databases, chunking strategies, and retrieval optimization, refer to the Local RAG 2026 guide. This comprehensive guide provides a complete overview of the best practices for building a local RAG pipeline.

Sources:
* [1] Qwen 3 14B or Gemma 3 12B instruction followers not specified
* [2] Local RAG 2026 guide
* [3] Ollama local RAG pipeline not specified
* [4] Pydantic AI agentic RAG not specified
* [5] Small LLMs customized teaching material generation not specified

## Manual checks (fill in after reading both outputs)

- [ ] Opened at least 3 URLs from the with-search output: do they exist, and do they support the claim beside them?
- [ ] Any URL in the without-search output? If yes, is it real, or invented?
- [ ] Specific facts (numbers, names, dates) present in the with-search output but missing from the other:
- [ ] Anything the Reviewer removed or flagged that you agree/disagree with:
- [ ] Overall improvement from adding search, in my own words:
