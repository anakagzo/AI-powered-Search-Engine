import os
import re
import json
import base64
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


class PostProcessor:
    """
    Handles the chunking of markdown document sections that do not have adequate headers or are too large.
    Post-processes markdown chunks with images using a vision-capable LLM to ensure RAG readiness.
    Handles table and image summarization and robust JSON extraction from LLM outputs.
    """

    def __init__(self, max_words: int = 500, api_key: str | None = None, model: str = "gpt-4.1-mini"):  
        """
        Initializes the PostProcessor with LLM client and parameters.
        args:
            max_words (int): Maximum words per chunk before splitting.
            api_key (str | None): OpenAI API key. If None, loads from environment.
            model (str): The LLM model to use for processing.
        """

        # Load environment variables for API access
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY") or api_key
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment")
        os.environ["OPENAI_API_KEY"] = api_key

        self.client = OpenAI(api_key=api_key)
        self.max_words = max_words
        self.model = model

        self.prompt = """
            You are an expert document post-processor for Retrieval-Augmented Generation (RAG).

            Your task is to structure Markdown content into retrieval-ready chunks.

            STRICT RULES:
            - Return STRICT JSON only. No explanations, no markdown fences.
            - Do NOT paraphrase.
            - Do NOT break tables or images.
            - If a table exists, keep it intact and add a concise summary AFTER it.
            - Preserve the Markdown text exactly as provided.
            - If text exceeds {MAX_WORDS}, split into coherent chunks.
            - Add a section title only if missing (maximum 1 sentence).

            IMAGE HANDLING:
            - Detect standard Markdown image syntax: ![alt text](path)
            - Extract ALL image paths exactly as written.
            - Do NOT invent or modify image paths.
            - If no images exist, return an empty list.

            For EACH returned chunk, output the following fields:
            - text 
            - section_title
            - chunk_type (one of: "text", "table/text", "image/text", table/image/text)
            - image_paths (array of strings, may be empty)

            Markdown input:
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
        

    def _extract_image_paths(self, markdown: str) -> list[str]:
        '''find all image paths in markdown using regex pattern'''
        image_regex = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        return image_regex.findall(markdown)
    
    def _inject_image_summary(self,markdown: str, image_path: str, summary: str) -> str:
        '''inject image summary (from LLM) after the image in markdown'''
        image_pattern = rf'(!\[[^\]]*\]\({re.escape(image_path)}\))'
        replacement = r'\1\n\n**Image summary:** ' + summary
        return re.sub(image_pattern, replacement, markdown, count=1)
    

    def summarize_image(self, image_path: str) -> str:
        """
        Uses a vision-capable LLM to summarise an image.

        args:
            image_path (str): The file path to the image.
        returns:
            str: The concise summary of the image content.
        """
        path = Path(image_path)
        if not path.exists():
            return "Image file not found."

        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode()

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Summarise the information shown in this image "
                                "clearly and concisely for semantic retrieval. "
                                "Do not speculate beyond visible content."
                            )
                        },
                        {
                            "type": "input_image",
                            "image_base64": image_b64
                        }
                    ]
                }
            ]
        )

        return response.output_text.strip()


    def process_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Robust LLM post-processing with JSON recovery. This includes:
        1- Chunk splitting if too large.
        2- Section title addition if missing.
        3- Table detection and summarization.
        4- Image path extraction and summarization.

        args:
            chunk (Dict[str, Any]): The markdown chunk with metadata.
        returns:
            List[Dict[str, Any]]: The list of refined chunks after processing.
        """
        markdown = chunk["text"]

        # Extract image paths and summarize each image
        image_paths = self._extract_image_paths(markdown)
        for image_path in image_paths:
            summary = self.summarize_image(image_path)
            markdown = self._inject_image_summary(markdown, image_path, summary)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only."},
                {"role": "user", "content": self.prompt.format(text=markdown, MAX_WORDS=self.max_words)}
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
                    else "image & text" if image_paths
                    else "text"
                )
            }]       
