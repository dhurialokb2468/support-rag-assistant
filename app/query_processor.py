import json
import re
import time
from typing import Any

from app.models import QueryMetadata, QueryRewriteTrace

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "reporting": {
        "reporting", "report", "reports", "export", "csv", "pdf", "dashboard", "analytics"
    },
    "authentication": {
        "authentication", "auth", "login", "password", "sso", "oauth", "token", "credentials"
    },
    "billing": {
        "billing", "invoice", "subscription", "payment", "plan", "charge"
    },
    "integrations": {
        "integrations", "integration", "salesforce", "webhook", "api", "slack", "zapier", "connector", "sync"
    },
    "permissions": {
        "permissions", "permission", "role", "roles", "rbac", "access", "unauthorized", "forbidden"
    },
    "user-management": {
        "user-management", "usermanagement", "user management", "users", "user", "account", "team", "organization", "provisioning"
    },
}

TECHNICAL_TERM_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,10}-\d{3,5}|[A-Z][a-zA-Z0-9_-]+|[a-z0-9_]+[-_][a-z0-9_-]+|[A-Z]{2,})\b"
)

AMBIGUOUS_PRONOUNS = {
    "it", "its", "that", "this", "they", "them", "these", "those"
}


class QueryProcessor:
    def extract_product(self, query: str) -> str | None:
        if re.search(r"\binsightflow\b", query, re.IGNORECASE):
            return "InsightFlow"
        return None

    def extract_version(self, query: str) -> str | None:
        pattern = r"(?<![A-Za-z0-9-])(?:version\s+|v)?(\d+\.\d+(?:\.\d+)?)\b"
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def extract_error_codes(self, query: str) -> list[str]:
        pattern = r"\b[A-Za-z]{2,10}-\d{3,5}\b"
        matches = re.findall(pattern, query)
        seen = set()
        error_codes = []
        for match in matches:
            code = match.upper()
            if code not in seen:
                seen.add(code)
                error_codes.append(code)
        return error_codes

    def extract_category(self, query: str) -> str | None:
        query_lower = query.lower()
        tokens = set(re.findall(r"\b[a-z0-9_-]+\b", query_lower))

        for cat in CATEGORY_KEYWORDS:
            if cat in tokens or cat in query_lower:
                return cat

        scores: dict[str, int] = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in tokens or kw in query_lower)
            if count > 0:
                scores[cat] = count

        if not scores:
            return None

        return max(scores, key=lambda k: scores[k])

    def extract_terms(
        self, query: str, product: str | None, version: str | None, error_codes: list[str]
    ) -> list[str]:
        terms: list[str] = []

        def add_term(term: str) -> None:
            if term and term not in terms:
                terms.append(term)

        if product:
            add_term(product)
        if version:
            add_term(version)
            add_term(f"v{version}")
            add_term(f"version {version}")
        for code in error_codes:
            add_term(code)

        raw_matches = TECHNICAL_TERM_PATTERN.findall(query)
        for match in raw_matches:
            if match.lower() not in {"the", "and", "for", "with", "this", "that"}:
                add_term(match)

        return terms

    def process(self, query: str) -> QueryMetadata:
        product = self.extract_product(query)
        version = self.extract_version(query)
        error_codes = self.extract_error_codes(query)
        category = self.extract_category(query)
        extracted_terms = self.extract_terms(query, product, version, error_codes)

        return QueryMetadata(
            product=product,
            version=version,
            category=category,
            error_codes=error_codes,
            extracted_terms=extracted_terms,
        )

    def extract_metadata(self, query: str) -> QueryMetadata:
        return self.process(query)


def extract_query_metadata(query: str) -> QueryMetadata:
    processor = QueryProcessor()
    return processor.process(query)


def needs_query_rewrite(
    question: str,
    conversation_state: Any | None = None,
) -> bool:
    tokens = set(re.findall(r"\b[a-z]+\b", question.lower()))

    # 1. Contains ambiguous pronouns
    if bool(tokens.intersection(AMBIGUOUS_PRONOUNS)):
        return True

    # 2. Question is very short
    words = [w for w in question.strip().split() if w]
    if len(words) <= 4:
        return True

    # 3. Context dependency with conversation_state
    if conversation_state:
        context_str = ""
        if hasattr(conversation_state, "build_rewrite_context"):
            context_str = conversation_state.build_rewrite_context()
        elif hasattr(conversation_state, "get_context_text"):
            context_str = conversation_state.get_context_text()
        elif isinstance(conversation_state, list):
            context_str = "\n".join(str(m.get("content", "")) for m in conversation_state)

        if context_str.strip():
            processor = QueryProcessor()
            q_meta = processor.process(question)

            # State has product/version/codes that question is missing
            s_ver = getattr(conversation_state, "product_version", None)
            s_prod = getattr(conversation_state, "product", None)
            s_codes = getattr(conversation_state, "error_codes", [])

            if (s_prod and not q_meta.product) or (s_ver and not q_meta.version) or (s_codes and not q_meta.error_codes):
                return True

            if any(question.lower().startswith(prefix) for prefix in ["why", "how", "what about", "and", "can i", "is it"]):
                return True

    return False


def rewrite_query(
    question: str,
    conversation_state: Any | None = None,
    generator: Any | None = None,
) -> tuple[str, QueryRewriteTrace]:
    started = time.perf_counter()

    if not needs_query_rewrite(question, conversation_state):
        trace = QueryRewriteTrace(
            original_question=question,
            rewritten_query=question,
            rewrite_used=False,
            rewrite_latency=0.0,
        )
        return question, trace

    if generator is None:
        try:
            from app.generator import OllamaGenerator
            gen_inst = OllamaGenerator()
            if gen_inst.is_available():
                generator = gen_inst
        except Exception:
            generator = None

    if generator is None:
        elapsed = time.perf_counter() - started
        trace = QueryRewriteTrace(
            original_question=question,
            rewritten_query=question,
            rewrite_used=False,
            rewrite_latency=elapsed,
        )
        return question, trace

    context_text = ""
    if hasattr(conversation_state, "build_rewrite_context"):
        context_text = conversation_state.build_rewrite_context()
    elif hasattr(conversation_state, "get_context_text"):
        context_text = conversation_state.get_context_text()
    elif isinstance(conversation_state, list):
        context_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in conversation_state
        )

    prompt = f"""
You are a search query reformulation module for InsightFlow support data.

Re-write the user's latest question into a standalone search query that incorporates necessary context from prior conversation.

Rules:
1. Do NOT answer the question.
2. Do NOT add new facts not present in the context or question.
3. Preserve all product names, version numbers, and error codes (e.g., InsightFlow, version 3.2, EXP-3204).
4. Output ONLY the single rewritten search query string. Do not output explanations, markdown formatting, or preamble.

Conversation History:
{context_text if context_text else 'None'}

User Question:
{question}

Standalone Search Query:
"""

    try:
        response, elapsed = generator.generate(prompt, temperature=0.0)
        cleaned_query = response.strip().strip('"').strip("'").strip()
        if not cleaned_query:
            cleaned_query = question

        trace = QueryRewriteTrace(
            original_question=question,
            rewritten_query=cleaned_query,
            rewrite_used=True,
            rewrite_latency=elapsed,
        )
        return cleaned_query, trace

    except Exception:
        elapsed = time.perf_counter() - started
        trace = QueryRewriteTrace(
            original_question=question,
            rewritten_query=question,
            rewrite_used=False,
            rewrite_latency=elapsed,
        )
        return question, trace


def generate_search_queries(
    standalone_query: str,
    generator: Any | None = None,
    max_queries: int = 3,
) -> list[str]:
    if not standalone_query or not standalone_query.strip():
        return [standalone_query] if standalone_query is not None else []

    clean_original = standalone_query.strip()

    if generator is None:
        try:
            from app.generator import OllamaGenerator
            gen_inst = OllamaGenerator()
            if gen_inst.is_available():
                generator = gen_inst
        except Exception:
            generator = None

    if generator is None:
        return [clean_original]

    prompt = f"""
You are a search query expansion module for InsightFlow support data.

Generate multi-query search variations for the given standalone search query.

Requirements:
1. Generate no more than {max_queries} queries in total.
2. Include:
   - The original standalone query
   - One symptom-oriented variation (describing user symptoms/behavior)
   - One documentation-style variation (phrased like official documentation or guide titles)
3. Preserve EXACT:
   - Error codes (e.g., EXP-3204, AUTH-101)
   - Version numbers (e.g., 3.2, v3.2)
   - Product names (e.g., InsightFlow)
   - Integration names (e.g., Salesforce, Slack, Zapier)
4. Do NOT introduce unsupported possible causes or unmentioned features/errors.
5. Return ONLY a structured JSON object in the following format:
{{
  "queries": [
    "<original query>",
    "<symptom-oriented variation>",
    "<documentation-style variation>"
  ]
}}

Standalone query:
{clean_original}
"""

    try:
        data = None
        if hasattr(generator, "generate_json"):
            res = generator.generate_json(prompt)
            data = res[0] if isinstance(res, tuple) else res
        elif hasattr(generator, "generate"):
            res = generator.generate(prompt)
            raw_text = res[0] if isinstance(res, tuple) else res
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
                cleaned = re.sub(r"```$", "", cleaned).strip()
            data = json.loads(cleaned)

        raw_queries: list[str] = []
        if isinstance(data, dict):
            if "queries" in data and isinstance(data["queries"], list):
                raw_queries = [str(q).strip() for q in data["queries"] if str(q).strip()]
            else:
                for k in ["original", "standalone", "symptom", "symptom_oriented", "documentation", "documentation_style"]:
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        raw_queries.append(data[k].strip())
        elif isinstance(data, list):
            raw_queries = [str(q).strip() for q in data if str(q).strip()]

        if not raw_queries:
            return [clean_original]

        final_queries: list[str] = []
        if clean_original not in raw_queries:
            final_queries.append(clean_original)

        for q in raw_queries:
            if q not in final_queries:
                final_queries.append(q)

        return final_queries[:max_queries]

    except Exception:
        return [clean_original]

