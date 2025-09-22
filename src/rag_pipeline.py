"""
RAG Pipeline Orchestrator

Goal: Orchestrate the complete RAG pipeline combining retrieval and generation.
This is the main pipeline that coordinates between retriever and generator.

Responsibilities:
- Coordinate retrieval and generation steps
- Process user queries end-to-end
- Handle error cases and fallbacks
- Manage conversation context
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from typing import List, Dict, Any, Generator

from generator import MovieGenerator
from retriever import MovieRetriever

class RAGPipeline:
    """
    Orchestrator for a RAG pipeline with conversation memory.
    Supports streaming generation.
    """

    def __init__(self, max_context_tokens: int = 2048):
        self.retriever = MovieRetriever()
        self.generator = MovieGenerator()
        self.max_context_tokens = max_context_tokens
        self.conversation_history: List[Dict[str, str]] = []  # {"user": str, "assistant": str}

    def process_query(self, user_query: str,top_k: int = 10, stream: bool = False) -> Any:
        """
        End-to-end RAG: retrieve contexts, include conversation memory, generate response.
        Supports streaming mode.
        """
        try:
            # Step 1: Retrieve contexts
            contexts = self.retriever.retrieve(user_query, top_k=top_k)

            # Step 2: Include conversation memory
            conversation_context = self._build_conversation_context()
            all_contexts = conversation_context + contexts

            # Step 3: Truncate to fit token budget
            truncated_contexts = self.truncate_contexts(
                all_contexts, user_query, max_tokens=self.max_context_tokens
            )

            # Step 4: Generate response
            response = self.generator.generate_with_context(user_query, truncated_contexts, stream)
            self.conversation_history.append({"user": user_query, "assistant": response})
            return response

        except Exception as e:
            return f"Error processing query: {e}"

    def _build_conversation_context(self) -> List[str]:
        """Format past conversation turns into strings for inclusion in the prompt."""
        formatted_history = []
        for turn in self.conversation_history[-5:]:  # keep last 5 turns
            formatted_history.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        return formatted_history

    def truncate_contexts(self, contexts: List[str], user_query: str, max_tokens: int = 2048) -> List[str]:
        """Truncate contexts + conversation memory to fit token budget."""
        if not contexts:
            return []
        
        query_tokens = int(len(user_query.split()) * 1.3)
        budget = max_tokens - query_tokens

        truncated = []
        used_tokens = 0
        for ctx in contexts:
            tokens = int(len(ctx.split()) * 1.3)
            if used_tokens + tokens <= budget:
                truncated.append(ctx)
                used_tokens += tokens
            else:
                remaining = budget - used_tokens
                if remaining > 0:
                    words = ctx.split()
                    approx_words = int(remaining / 1.3)
                    truncated.append(" ".join(words[:approx_words]) + " …")
                break
        return truncated

    def reset_conversation(self):
        """Clear conversation memory."""
        self.conversation_history = []
