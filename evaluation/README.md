# InsightFlow RAG Evaluation Benchmark Dataset

This evaluation dataset contains **60 carefully designed evaluation test cases** (`evaluation/questions.json`) to benchmark the performance of the InsightFlow Support RAG Assistant.

## Benchmark Schema

Each test case in `evaluation/questions.json` conforms to the following schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Unique question ID (e.g. `q01`, `q60`) |
| `question` | string | The query text evaluated |
| `conversation_context` | string \| null | Optional prior turn context for query rewriting |
| `expected_source_ids` | list[string] | Canonical document IDs expected to support the answer |
| `expected_answer_points` | list[string] | Key factual points required in the answer |
| `category` | string \| null | Target metadata category (`reporting`, `integrations`, `authentication`) |
| `version` | string \| null | Target product version (`3.2`, `3.1`, `all`, null) |
| `answerable` | boolean | True if answerable from knowledge base, False if unanswerable |
| `expected_abstention` | boolean | True if system should trigger explicit abstention |
| `expected_escalation` | boolean | True if system should trigger human escalation |
| `difficulty` | string | Difficulty rating (`easy`, `medium`, `hard`) |
| `test_type` | string | One of 15 specialized RAG evaluation categories |

---

## 15 Evaluation Test Types

1. **`direct_factual`**: Straightforward questions matching document text directly.
2. **`paraphrased`**: Synonymous/phrased queries testing semantic vector retrieval.
3. **`exact_error_code`**: Specific error code queries (e.g. `EXP-3204`).
4. **`version_specific`**: Queries testing version filtering (e.g. `3.2` vs `3.1`).
5. **`multi_source`**: Queries requiring synthesis across multiple documents (e.g. Release Notes + Known Issues).
6. **`conversational_followup`**: Anaphoric follow-up queries depending on `conversation_context`.
7. **`ambiguous_question`**: Underspecified queries testing abstention or clarification logic.
8. **`unsupported_question`**: Out-of-domain questions testing knowledge base boundary detection.
9. **`future_plan`**: Questions about undocumented future plans/roadmaps testing abstention.
10. **`conflicting_source`**: Queries matching conflicting documents testing priority resolution.
11. **`outdated_source_trap`**: Traps testing version freshness and document type precedence.
12. **`metadata_filtering`**: Queries testing metadata constraints (`category`, `version`).
13. **`bm25_technical`**: Exact token queries testing BM25 keyword matching.
14. **`query_rewriting`**: Context-dependent queries testing LLM standalone query reformulation.
15. **`parent_child_context`**: Queries testing parent section chunk resolution.

---

## Running Evaluation

Run the automated evaluation runner CLI script:

```bash
python -m scripts.evaluate --questions evaluation/questions.json
```
