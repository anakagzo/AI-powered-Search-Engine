import re
from typing import List, Dict, Any
from datetime import datetime
from app.utils.hashing import deterministic_hash
from app.llm.post_processor import PostProcessor


class HierarchicalChunker:
    """
    Chunks markdown documents hierarchically based on headers (H1, H2, H3).
    Applies LLM post-processing for unstructured content like tables and images, and 
    further splits large chunk sections.
    """

    def __init__(self, max_words: int = 500): 
        """
        Initialize the HierarchicalChunker.

        Args:
            max_words (int): Maximum number of allowed words per chunk.
        """
        self.max_words = max_words


    def word_count(self, text: str) -> int:
        """Count the number of words in a chunk."""
        return len(text.split())


    def _find_headers(self, markdown: str, pattern: str):
        """
        Find all header matches in the markdown text 
        starting with H1, followed by H2 and H3 (when applicable) using the provided pattern.
        Returns:
        - list of sections matching the header pattern.
        """
        return list(re.finditer(pattern, markdown, flags=re.MULTILINE))


    def _extract_header(self, section: str):
        """
        Extracts the header level and title from a markdown section.    
        Returns:
        - header_level (str | None): The level of the header (e.g., '#', '##', '###').
        - header_title (str | None): The title text of the header.
        """
        m = re.match(r"^(#{1,3})\s+(.*)", section)
        return (m.group(1), m.group(2).strip()) if m else (None, None)


    def _split_with_intro(self, markdown: str, pattern: str):
        """
        Splits a header section into subsections while preserving text before its first subheader.

        Returns:
        - intro_text (str | None)
        - list of subsections
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


    def contains_table(self, markdown: str) -> bool:
        """
        Detects if a markdown string contains a table.

        Args:
            markdown (str): The markdown text to analyze.

        Returns:
            bool: True if a table is detected, False otherwise.
        """
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


    def contains_image(self, markdown: str) -> bool:
        """
        Detects if a markdown string contains an image. 
        
        Args:
            markdown (str): The markdown text to analyze.   
        Returns:
            bool: True if an image is detected, False otherwise.
        """
        return bool(re.search(r"!\[.*?\]\(.*?\)", markdown))


    def _contains_table_or_image(self, markdown: str) -> bool:
        """
        Checks if a markdown section/chunk contains either a table or an image.
        This unified detection helps in deciding if LLM post processing is needed.
        """
        return self.contains_table(markdown) or self.contains_image(markdown)


    def chunk_markdown_by_headers(
        self,
        markdown: str,
        source_file: str = "uploaded_docx"
    ) -> List[Dict[str, Any]]:
        """
        Chunks the markdown document hierarchically based on headers (H1, H2, H3).
        Large sections or those containing tables/images are marked for LLM post-processing.  

        Note: breaking of sections into smaller chunks is only done if they exceed max_words limit. 
            Documents without any headers are treated as single chunks and marked for LLM processing.

        Args:
            markdown (str): The markdown content to be chunked.
            source_file (str): The source filename for metadata purposes.   
        Returns:
            List[Dict[str, Any]]: A list of chunk dictionaries with text and metadata.  
        """

        chunks = []  # List to hold the resulting chunks
        idx = 0 # takes count of chunks created, used as index for creating chunk unique IDs

        # step 1. check for H1 headers in the markdown document - Split by H1
        h1_matches = self._find_headers(markdown, r"^#\s+")

        if not h1_matches:
            # No H1 headers found, treat entire document as a single chunk, 
            # then mark document for LLM processing - chunking by LLM will be done later
           
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
        
        # step 2. Process intro text before first H1 (if any) - treat as separate chunk
        # this is a case where document has intro text before first H1 header 
        intro = markdown[:h1_matches[0].start()].strip()
        if intro:
            # mark intro chunk for LLM processing if it contains table/image or exceeds max words
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
            idx += 1 # increment chunk index

        # step 3. Process each H1 section, if too large, split further by H2 
        for i, h1_match in enumerate(h1_matches):
            start = h1_match.start()
            end = h1_matches[i + 1].start() if i + 1 < len(h1_matches) else len(markdown)
            h1_section = markdown[start:end].strip()

            _, h1_title = self._extract_header(h1_section)
            wc1 = self.word_count(h1_section)

            # check if section contains table or image
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

            # split large H1 section by H2
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

            # step 4. Process each H2 section, if too large, split further by H3
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

                # split large H2 section by H3
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

                # step 5. Process each H3 section, if too large, mark for LLM processing
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


    def finalize_chunks(self, initial_chunks: List[Dict[str, Any]], 
                        source_file: str="uploaded_docx")-> List[Dict[str, Any]]:
        """
        Finalizes chunks by applying LLM post-processing where needed.

        Args:
            initial_chunks (List[Dict[str, Any]]): The list of initial chunks with metadata
            source_file (str): The source filename for metadata purposes.
        Returns:
            List[Dict[str, Any]]: The finalized list of chunks ready for RAG ingestion with retrieval optimized metadata
        """
        final_chunks = []
        idx = 0
        today = datetime.now().date().isoformat()

        for each_chunk in initial_chunks:
            meta = each_chunk["metadata"]

            # If no LLM processing is needed, add chunk as is
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

            # Append all refined chunks after LLM processing and update metadata
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