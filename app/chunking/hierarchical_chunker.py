import re
from typing import List, Dict, Any
from datetime import datetime
from app.utils.hashing import deterministic_hash
from app.llm.post_processor import PostProcessor


class HierarchicalChunker:
    """
    Chunks markdown documents hierarchically based on headers (H1, H2, H3).
    Applies LLM post-processing for large or structured content.
    """

    def __init__(self, max_words: int = 500):    
        self.max_words = max_words

    def word_count(self, text: str) -> int:
        return len(text.split())

    def _find_headers(self, markdown: str, pattern: str):
        return list(re.finditer(pattern, markdown, flags=re.MULTILINE))


    def _extract_header(self, section: str):
        m = re.match(r"^(#{1,3})\s+(.*)", section)
        return (m.group(1), m.group(2).strip()) if m else (None, None)


    def _split_with_intro(self, markdown: str, pattern: str):
        """
        Splits markdown while preserving text before first subheader.

        Returns:
        - intro_text (str | None)
        - list of subsection strings
        """
        matches = list(re.finditer(pattern, markdown, flags=re.MULTILINE))

        if not matches:
            return None, []

        intro_text = markdown[:matches[0].start()].strip()
        sections = []

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            sections.append(markdown[start:end].strip())

        return intro_text, sections


    # =========================
    # strict table detection (miss-averse)
    # =========================
    def contains_table(self, markdown: str) -> bool:
        lines = markdown.splitlines()

        pipe_like_lines = 0
        separator_found = False

        for line in lines:
            line = line.strip()

            # Pipe-based row
            if "|" in line and len(line.split("|")) >= 3:
                pipe_like_lines += 1

            # Header separator row (Pandoc / GitHub style)
            if re.match(r"^\s*\|?\s*[-:]{3,}.*[-:]{3,}\s*\|?\s*$", line):
                separator_found = True

            # Pandoc grid table markers
            if re.match(r"^\s*\+[-=]+\+\s*$", line):
                return True

        # Conservative rule:
        # - at least 2 pipe-like rows OR
        # - explicit separator row
        return pipe_like_lines >= 2 or separator_found


    # =========================
    # NEW: strict image detection
    # =========================
    def contains_image(self, markdown: str) -> bool:
        return bool(re.search(r"!\[.*?\]\(.*?\)", markdown))


    # =========================
    # NEW: unified detector
    # =========================
    def _contains_table_or_image(self, markdown: str) -> bool:
        return self.contains_table(markdown) or self.contains_image(markdown)


    def chunk_markdown_by_headers(
        self,
        markdown: str,
        source_file: str = "uploaded_docx"
    ) -> List[Dict[str, Any]]:

        chunks = []
        idx = 0

        # 1️⃣ Text before first H1
        h1_matches = self._find_headers(markdown, r"^#\s+")

        if not h1_matches:
            # table/image detection applied here
            needs_llm = self._contains_table_or_image(markdown) or self.word_count(markdown) > self.max_words
            chunks.append({
                "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                "text": markdown.strip(),
                "metadata": {
                    "headers": {},
                    "word_count": self.word_count(markdown),
                    "needs_llm": True,
                    "contains_table": self.contains_table(markdown),
                    "contains_image": self.contains_image(markdown)
                }
            })
            return chunks

        intro = markdown[:h1_matches[0].start()].strip()
        if intro:
            needs_llm = self._contains_table_or_image(intro) or self.word_count(intro) > self.max_words

            chunks.append({
                "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                "text": intro,
                "metadata": {
                    "headers": {},
                    "word_count": self.word_count(intro),
                    "needs_llm": True,
                    "contains_table": self.contains_table(intro),     
                    "contains_image": self.contains_image(intro)     
                }
            })
            idx += 1

        # 2️⃣ Process each H1 section
        for i, h1_match in enumerate(h1_matches):
            start = h1_match.start()
            end = h1_matches[i + 1].start() if i + 1 < len(h1_matches) else len(markdown)
            h1_section = markdown[start:end].strip()

            _, h1_title = self._extract_header(h1_section)
            wc1 = self.word_count(h1_section)

            # NEW: unified detection
            has_structured_content = self._contains_table_or_image(h1_section)
            if wc1 <= self.max_words:
                chunks.append({
                    "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                    "text": h1_section,
                    "metadata": {
                        "headers": {"h1": h1_title},
                        "word_count": wc1,
                        "needs_llm": has_structured_content,   
                        "contains_table": self.contains_table(h1_section),  
                        "contains_image": self.contains_image(h1_section)   
                    }
                })
                idx += 1
                continue

            # 3️⃣ Split H1 → H2
            h1_intro, h2_sections = self._split_with_intro(h1_section, r"^##\s+")
            if h1_intro:
                wc_intro = self.word_count(h1_intro)
                needs_llm = self._contains_table_or_image(h1_intro) or wc_intro > self.max_words

                chunks.append({
                    "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                    "text": h1_intro,
                    "metadata": {
                        "headers": {"h1": h1_title},
                        "word_count": wc_intro,
                        "needs_llm": needs_llm,
                        "contains_table": self.contains_table(h1_intro),   
                        "contains_image": self.contains_image(h1_intro)   
                    }
                })
                idx += 1

            for h2 in h2_sections:
                _, h2_title = self._extract_header(h2)
                wc2 = self.word_count(h2)
                has_structured_content = self._contains_table_or_image(h2)

                if wc2 <= self.max_words:
                    chunks.append({
                        "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                        "text": h2,
                        "metadata": {
                            "headers": {"h1": h1_title, "h2": h2_title},
                            "word_count": wc2,
                            "needs_llm": has_structured_content,  
                            "contains_table": self.contains_table(h2), 
                            "contains_image": self.contains_image(h2)  
                        }
                    })
                    idx += 1
                    continue

                # 4️⃣ Split H2 → H3
                h2_intro, h3_sections = self._split_with_intro(h2, r"^###\s+")

                if h2_intro:
                    wc2i = self.word_count(h2_intro)
                    needs_llm = self._contains_table_or_image(h2_intro) or wc2i > self.max_words
                    chunks.append({
                        "chunk_id": f"{source_file}::chunk_{idx}",
                        "text": h2_intro,
                        "metadata": {
                            "headers": {"h1": h1_title, "h2": h2_title},
                            "word_count": wc2i,
                            "needs_llm": needs_llm,
                            "contains_table": self.contains_table(h2_intro),
                            "contains_image": self.contains_image(h2_intro) 
                        }
                    })
                    idx += 1

                for h3 in h3_sections:
                    _, h3_title = self._extract_header(h3)
                    wc3 = self.word_count(h3)

                    needs_llm = self._contains_table_or_image(h3) or wc3 > self.max_words

                    chunks.append({
                        "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                        "text": h3,
                        "metadata": {
                            "level": "h3",
                            "headers": {
                                "h1": h1_title,
                                "h2": h2_title,
                                "h3": h3_title
                            },
                            "word_count": wc3,
                            "needs_llm": needs_llm,
                            "contains_table": self.contains_table(h3),  
                            "contains_image": self.contains_image(h3)   
                        }
                    })
                    idx += 1

        return chunks



    # ============================================================
    # FINAL NORMALIZATION (MERGE BOTH STAGES)
    # ============================================================

    def finalize_chunks(self, initial_chunks: List[Dict[str, Any]], 
                        source_file: str="uploaded_docx")-> List[Dict[str, Any]]:
        """
        Produces final production-ready chunks with full metadata.
        """
        final_chunks = []
        idx = 0
        today = datetime.now().date().isoformat()

        for each_chunk in initial_chunks:
            #print(chunk)
            meta = each_chunk["metadata"]

            if not meta["needs_llm"]:
                final_chunks.append({
                    "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                    "content": each_chunk["text"],
                    "metadata": {
                        "source": source_file,
                        "chunk_type": "text",
                        "heading": meta.get("headers", {}).get("h1"),
                        "sub_heading": meta.get("headers", {}).get("h2"),
                        "section_title": meta.get("headers", {}).get("h3") or
                            meta.get("headers", {}).get("h2")
                            or meta.get("headers", {}).get("h1"),
                        "date": today
                    }
                })
                idx += 1
                continue

            # LLM post-processing
            refined = PostProcessor(max_words=self.max_words).process_chunk(each_chunk)

            for r in refined:
                final_chunks.append({
                    "chunk_id": deterministic_hash(f"{source_file}::chunk_{idx}"),
                    "content": r["text"],
                    "metadata": {
                        "source": source_file,
                        "chunk_type": r["chunk_type"],
                        "heading": meta.get("headers", {}).get("h1")
                            or r["section_title"],
                        "sub_heading": meta.get("headers", {}).get("h2"),
                        "section_title": meta.get("headers", {}).get("h2") or r["section_title"],
                        "image_paths": r["image_paths"] if r["image_paths"] else None,
                        "date": today
                    }
                })
                idx += 1

        return final_chunks