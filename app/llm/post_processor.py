import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


class PostProcessor:
    """
    LLM-based post-processing utilities for document chunks.
    """

    def __init__(self, max_words: int = 500, api_key: str | None = None):  
        # Load environment variables for API access
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY") or api_key
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment")
        os.environ["OPENAI_API_KEY"] = api_key

        self.client = OpenAI(api_key=api_key)

        self.max_words = max_words

        self.prompt = """
        You are an expert document post-processor.

        Rules:
        - Do NOT paraphrase.
        - Do NOT break tables or images.
        - If text exceeds {MAX_WORDS}, split into coherent chunks.
        - If a table or image exists, keep it intact and add a concise summary AFTER it.
        - Add a section title if missing (≤ 1 sentence).

        Return STRICT JSON only.

        For each chunk return:
        - text
        - section_title
        - chunk_type ("text", "table & text", "image & text")

        Markdown:
        ----------------
        {text}
        ----------------
        """

    def safe_json_loads(self, text: str):
        """
        Extracts and parses JSON safely from LLM output.
        """
        if not text:
            raise ValueError("LLM returned empty response")

        # Remove code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text)
            text = re.sub(r"```$", "", text)
            text = text.strip()

        # Find first JSON object or array
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")

        return json.loads(match.group(1))


    def process_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Robust LLM post-processing with JSON recovery.
        """
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Return STRICT JSON only."},
                {"role": "user", "content": self.prompt.format(text=chunk["text"], MAX_WORDS=self.max_words)}
            ],
            temperature=0.1
        )

        raw_output = response.choices[0].message.content

        try:
            return self.safe_json_loads(raw_output)
        except Exception as e:
            # FAIL SAFE: return original chunk unchanged
            print("⚠️ LLM JSON PARSE FAILED:", e)
            return [{
                "text": chunk["text"],
                "section_title": chunk["metadata"].get("headers", {}).get("h3") or 
                chunk["metadata"].get("headers", {}).get("h2")
                    or chunk["metadata"].get("headers", {}).get("h1")
                    or "Untitled Section",
                "chunk_type": (
                    "table & text" if chunk["metadata"].get("contains_table")
                    else "image & text" if chunk["metadata"].get("contains_image")
                    else "text"
                )
            }]

