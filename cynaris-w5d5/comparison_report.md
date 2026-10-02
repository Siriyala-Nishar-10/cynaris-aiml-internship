# W5D5: llama3.2:3b vs qwen2.5:3b

Same retrieved context and system prompt for both models (min_sim=0).

## Q: What is the main topic of this document?

| Model       | Seconds | Tokens | Tokens/s |
| ----------- | ------- | ------ | -------- |
| llama3.2:3b | 8.51    | 34     | 4.0      |
| qwen2.5:3b  | 44.9    | 46     | 1.0      |

**llama3.2:3b**

> The main topic of this document is streamlining the workload within Human Resources (HR) departments through the development of intelligent banking assistants. [sample.pdf p.1]

**qwen2.5:3b**

> The main topic of this document is the development of an intelligent banking assistant using RAG systems to streamline HR workload by efficiently handling queries related to internal organizational policies and legal requirements, despite the unstructured nature of normative documents.

## Q: What language is the data in, and what are the two phases of the research?

| Model       | Seconds | Tokens | Tokens/s |
| ----------- | ------- | ------ | -------- |
| llama3.2:3b | 28.82   | 45     | 1.6      |
| qwen2.5:3b  | 22.63   | 36     | 1.6      |

**llama3.2:3b**

> The data is in natural language. The research is executed in two main phases: an initial phase with structured Excel question-answer pairs, and a more complex phase with unstructured PDF documents. [sample.pdf p.5]

**qwen2.5:3b**

> The data is in natural language. The research has two main phases: an initial phase with structured Excel question-answer pairs, and a more complex phase with unstructured PDF documents.

## Q: Who won the 2010 FIFA World Cup?

| Model       | Seconds | Tokens | Tokens/s |
| ----------- | ------- | ------ | -------- |
| llama3.2:3b | 18.71   | 42     | 2.2      |
| qwen2.5:3b  | 15.48   | 10     | 0.6      |

**llama3.2:3b**

> I don't know based on the documents. [sample.pdf p.5] does not mention the 2010 FIFA World Cup, and the other documents do not provide any relevant information on this topic.

**qwen2.5:3b**

> I don't know based on the documents.
