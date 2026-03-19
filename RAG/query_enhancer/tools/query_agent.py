import sys
from pathlib import Path
from openai import OpenAI
import re
import json

try:
    from config import OPENROUTER_API_KEY, OPENROUTER_SITE_NAME, OPENROUTER_BASE_URL, OPENROUTER_SITE_URL
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import OPENROUTER_API_KEY, OPENROUTER_SITE_NAME, OPENROUTER_BASE_URL, OPENROUTER_SITE_URL

_ENHANCE_PROMPT = """\
You are a search-query expert for a RAG retrieval system working on technical \
and academic documents.

Given the user question below, output a single JSON object with these keys:

"sub_queries"  — array of exactly {n} short, self-contained questions that \
together cover every important aspect of the original question \
(e.g. definition, mechanism, evidence, comparison, limitations). \
Each must be independently searchable.

"hyde"         — a 2-3 sentence hypothetical passage written as if it appeared \
in a relevant technical or academic document that directly answers the question. \
Use precise domain vocabulary. This passage will be embedded and compared against \
document chunks, so it must sound like document text, not a question.

"step_back"    — ONE broader question that captures the underlying principle \
or concept behind the original question.

Rules:
- Output ONLY valid JSON. No markdown fences, no preamble, no extra keys.
- "sub_queries" → array of strings
- "hyde"        → single string
- "step_back"   → single string

Example:
{{
    "sub_queries": ["What is X?", "How does X improve Y?", "What are X's limitations?"],
    "hyde": "X is a technique that achieves Y through Z. Empirical results show ...",
    "step_back": "What are the general principles of X-based approaches?"
}}

User question: {query}
"""


class QueryEnhancer:
    def __init__(
        self,
        n_subqueries=3,
        use_hyde=True,
        use_stepback=True,
        default_model="x-ai/grok-4.1-fast"
    ):
        self.api_key = OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Get one at https://openrouter.ai")

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )
        
        # Optional headers for rankings
        self.site_url = OPENROUTER_SITE_URL
        self.site_name = OPENROUTER_SITE_NAME
        self.n_subqueries = n_subqueries
        self.use_hyde = use_hyde
        self.use_stepback = use_stepback
        self.default_model = default_model

    def enhance(self, query):
        parsed = self._call_agent(query)
        queries = [query]

        for q in parsed.get("sub_queries", []):
            q = str(q).strip()
            if q:
                queries.append(q)

        if self.use_hyde:
            hyde = parsed.get("hyde", "").strip()
            if hyde:
                queries.append(hyde)

        if self.use_stepback:
            stepback = parsed.get("step_back", "").strip()
            if stepback:
                queries.append(stepback)

        seen, unique = set(), []
        for q in queries:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                unique.append(q)

        self._print_summary(unique)
        return unique

    def _call_agent(self, query, model=None, temperature=0.7, max_tokens=1000, top_p=1):
        """Call OpenRouter API and return parsed JSON response"""
        
        # Use provided model or default
        model_id = model or self.default_model
        
        prompt = _ENHANCE_PROMPT.format(n=self.n_subqueries, query=query)
        
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            # Get the response text correctly
            raw = response.choices[0].message.content.strip()
            
            # Remove markdown code fences if present
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

            # Extract JSON object
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            # Remove trailing commas (common LLM mistake)
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            
            parsed = json.loads(raw)
            
            # Validate the structure
            if not isinstance(parsed.get("sub_queries"), list):
                print("[QueryEnhancer] Invalid format: 'sub_queries' must be a list")
                return {}
            if not isinstance(parsed.get("hyde"), str):
                print("[QueryEnhancer] Invalid format: 'hyde' must be a string")
                return {}
            if not isinstance(parsed.get("step_back"), str):
                print("[QueryEnhancer] Invalid format: 'step_back' must be a string")
                return {}
            
            return parsed
        
        except json.JSONDecodeError as e:
            print(f"[QueryEnhancer] JSON parse error ({e}) — falling back to original query")
            print(f"[QueryEnhancer] Raw response: {raw[:200]}...")
            return {}
        except Exception as e:
            print(f"[QueryEnhancer] API call failed ({e}) — falling back to original query")
            return {}

    def _print_summary(self, queries):
        labels = ["original", *[f"sub-query {i}" for i in range(1, self.n_subqueries + 1)]]
        if self.use_hyde:
            labels.append("HyDE")
        if self.use_stepback:
            labels.append("step-back")

        print(f"[QueryEnhancer] {len(queries)} queries sent to retriever:")
        for i, q in enumerate(queries):
            label = labels[i] if i < len(labels) else f"query {i}"
            preview = q[:90] + "..." if len(q) > 90 else q
            print(f"  [{label}] {preview}")
        print()


if __name__ == "__main__":
    enhancer = QueryEnhancer(n_subqueries=3, use_hyde=True, use_stepback=True)
    test_q = "What is beta-VAE?"
    results = enhancer.enhance(test_q)

    print("=" * 60)
    for i, q in enumerate(results):
        label = "original" if i == 0 else f"query {i}"
        print(f"  {i + 1}. [{label}] {q}")
