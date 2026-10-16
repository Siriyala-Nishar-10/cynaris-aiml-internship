# W5D1: Local LLM Comparison Report

Generated: 2026-09-28 11:11  
Models: llama3.2:3b, qwen2.5:3b  
System prompt: `You are a helpful assistant. Be accurate and concise.`  
Options: `{'temperature': 0.2, 'seed': 42, 'num_predict': 300}`

> Automatic checks are simple substring/line-count tests -- a fast objective
> signal, not a full quality judgement. See the full responses below.

## Summary

| Question                 | Check                                  | llama3.2:3b check | llama3.2:3b tok/s | qwen2.5:3b check | qwen2.5:3b tok/s |
| ------------------------ | -------------------------------------- | ----------------- | ----------------- | ---------------- | ---------------- |
| Q1 factual               | mentions 'Nehru' and '1946'            | FAIL              | 13.2              | FAIL             | 13.6             |
| Q2 reasoning             | answer contains 80                     | PASS              | 12.3              | PASS             | 12.0             |
| Q3 instruction-following | exactly 3 bullets, each under 20 words | PASS              | 12.6              | PASS             | 12.9             |

## Full responses

### Q1 factual

**Prompt:** Who wrote 'The Discovery of India', and in what year was it first published?

**llama3.2:3b** (check: FAIL, 15 words, 13.2 tok/s)

> "The Discovery of India" was written by Mahatma Gandhi. It was first published in 1927.

**qwen2.5:3b** (check: FAIL, 29 words, 13.6 tok/s)

> 'My Life' (also known as 'The Discovery of India') was written by Indira Gandhi, the first and only female Prime Minister of India. It was first published in 1983.

### Q2 reasoning

**Prompt:** A train travels 60 km in 45 minutes. What is its average speed in km/h?

**llama3.2:3b** (check: PASS, 53 words, 12.3 tok/s)

> To find the average speed, we need to convert the time from minutes to hours.
>
> 45 minutes = 45/60 = 0.75 hours
>
> Now, we can use the formula:
>
> Average Speed = Total Distance / Total Time
> = 60 km / 0.75 hours
> = 80 km/h
>
> So, the train's average speed is 80 km/h.

**qwen2.5:3b** (check: PASS, 86 words, 12.0 tok/s)

> To find the average speed in km/h, we use the formula:
>
> \[ \text{Average Speed} = \frac{\text{Distance}}{\text{Time}} \]
>
> Given that the train travels 60 km in 45 minutes, we first need to convert the time into hours since speed is measured in km/h.
>
> \[ 45 \text{ minutes} = \frac{45}{60} \text{ hours} = 0.75 \text{ hours} \]
>
> Now, we can calculate the average speed:
>
> \[ \text{Average Speed} = \frac{60 \text{ km}}{0.75 \text{ hours}} = 80 \text{ km/h} \]
>
> So, the average speed of the train is 80 km/h.

### Q3 instruction-following

**Prompt:** Explain what overfitting is in exactly 3 bullet points, each under 20 words.

**llama3.2:3b** (check: PASS, 50 words, 12.6 tok/s)

> Here are three bullet points explaining overfitting:
>
> • Model is too complex and fits noise in training data, rather than the underlying patterns.
> • Model performs well on training data, but poorly on new, unseen data.
> • Model is too specialized to the training data, losing its ability to generalize.

**qwen2.5:3b** (check: PASS, 33 words, 12.9 tok/s)

> - Overfitting occurs when a model learns training data too well, including noise.
> - Leads to poor generalization on unseen data, worsening performance.
> - Can be mitigated by techniques like cross-validation and regularization.

## My observation

- **Accuracy / hallucination:** which model got the facts right? Did either invent details?
  Both models failed the factual question. Llama 3.2 incorrectly attributed The Discovery of India to Mahatma Gandhi and gave the wrong publication year. Qwen 2.5 incorrectly attributed the book to Indira Gandhi and gave the wrong publication year. The correct answer is Jawaharlal Nehru, first published in 1946.

- **Reasoning:** did the arithmetic work show correct steps, or just a lucky number?
  Both models correctly calculated the train's average speed as 80 km/h. They showed the conversion of 45 minutes into 0.75 hours and applied the correct formula. Llama provided a more concise explanation, while Qwen included more mathematical formatting and detail.

- **Instruction following:** did each model respect 'exactly 3 bullets' and the word limit?
  Both models passed the automatic check for exactly three bullet points, each under 20 words. Qwen followed the requested format directly, while Llama added an introductory sentence before the bullet points.

- **Style and verbosity:** which answers were clearer, and which rambled?
  Llama 3.2 generally provided shorter and simpler explanations. Qwen 2.5 gave more detailed answers, particularly for the reasoning question. Qwen's response to the factual question was unnecessarily elaborate and contained fabricated information.

- **Speed:** tokens/sec difference, and whether it mattered in practice.
  Both models generated responses at approximately 12–14 tokens per second. Llama completed the reasoning question in 9.1 seconds, compared with Qwen's 16.9 seconds. Their generation speeds were similar, but Llama responded faster to the reasoning question in this test.

- **Overall:** which would I use for what, and why?
  Both models performed well on basic arithmetic and instruction-following tasks but failed the factual knowledge test. I would consider Llama 3.2 for concise responses and simple reasoning, while Qwen 2.5 may be useful when more detailed explanations are preferred. However, neither model should be relied upon for factual accuracy without verification.

- **Conclusion:**
  This comparison shows that local LLMs can handle basic reasoning and formatting tasks, but they can still hallucinate facts. Model selection should consider accuracy, response quality, speed, and the intended use case
