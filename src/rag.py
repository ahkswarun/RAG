from typing import List, Dict, Optional


class RAG:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def _format_history(self, chat_history: Optional[List[Dict]], max_turns: int = 6) -> str:
        if not chat_history:
            return ""

        recent = chat_history[-max_turns:]
        lines = []
        for message in recent:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content']}")
        return "\n".join(lines)

    def _rewrite_query(self, question: str, chat_history: Optional[List[Dict]]) -> str:
        """Turn follow-ups into a standalone search query."""
        if not chat_history:
            return question

        history_text = self._format_history(chat_history)
        prompt = f"""Rewrite the latest user question as a standalone search query for a document.

Rules:
1. If the latest question refers to earlier chat (e.g. "it", "that", "those", "explain simply"),
   resolve those references using chat history.
2. If the latest question is already clear and self-contained, return it as-is
   (you may lightly clean grammar).
3. If the latest question is clearly unrelated to the chat history and document topic
   (e.g. sports scores, personal info, unrelated trivia),
   return the latest question UNCHANGED. Do NOT force it into the document topic.
4. Return ONLY the search query text. No quotes, no explanation.

Chat history:
{history_text}

Latest question: {question}

Standalone search query:"""

        rewritten = self.llm.generate(prompt).strip().strip('"')
        return rewritten or question

    def ask(self, question: str, chat_history: Optional[List[Dict]] = None):
        search_query = self._rewrite_query(question, chat_history)
        chunks = self.retriever.retrieve(search_query, k=8)

        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the document.",
                "sources": [],
                "retrieved_chunks": [],
                "search_query": search_query,
            }

        context = "\n\n".join(
            f"[Page {chunk['page']}]\n{chunk['text']}"
            for chunk in chunks
        )
        history_text = self._format_history(chat_history)

        history_block = (
            f"Previous conversation:\n{history_text}\n\n"
            if history_text
            else ""
        )

        prompt = f"""Below are excerpts retrieved from a PDF.

Excerpts:
{context}

{history_block}Current user question: {question}

Write a helpful answer based on the excerpts and conversation.
Rules:
1. Always try to answer using the excerpts. For overview questions, summarize what the excerpts discuss.
2. Use chat history to understand follow-up questions, but ground facts in the excerpts.
3. Do not invent facts that are not supported by the excerpts.
4. Mention page numbers when useful.
5. Only if the excerpts are completely unrelated, say you could not find that information.

Answer:"""

        answer = self.llm.generate(prompt)
        sources = []
        seen_pages = set()

        for chunk in chunks:
            if chunk["page"] in seen_pages:
                continue

            seen_pages.add(chunk["page"])
            sources.append(
                {
                    "page": chunk["page"],
                    "preview": chunk["text"][:200].strip() + "...",
                    "distance": chunk.get("distance"),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
            "search_query": search_query,
        }
