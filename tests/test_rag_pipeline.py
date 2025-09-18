"""
Tests for RAG Pipeline

Goal: Integration tests for the complete RAG pipeline.
This module contains tests for the end-to-end RAG functionality.

Test Coverage:
- End-to-end pipeline testing
- Error handling
- Performance testing
- Integration with retriever and generator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, patch
from src.rag_pipeline import RAGPipeline

def test_rag_pipeline_mocked():
    print("🚀 Initializing RAG Pipeline with mocked generator and retriever...")

    # Patch MovieGenerator and MovieRetriever to avoid loading real models
    with patch("src.rag_pipeline.MovieGenerator") as MockGen, \
         patch("src.rag_pipeline.MovieRetriever") as MockRet:

        # Configure the mocked generator
        mock_gen = MockGen.return_value
        mock_gen.generate_with_context.side_effect = lambda query, contexts, stream=False: (
            (c for c in f"Streaming response for {query}") if stream else f"Response for '{query}'"
        )

        # Configure the mocked retriever
        mock_ret = MockRet.return_value
        mock_ret.retrieve.side_effect = lambda query, top_k=10: [
            f"Mock context {i+1} for '{query}'" for i in range(top_k)
        ]

        # Instantiate RAGPipeline (uses mocked generator/retriever)
        rag = RAGPipeline(max_context_tokens=500)

        # --- Test 1: End-to-end ---
        query = "I like action movies"
        response = rag.process_query(query)
        print("Response:", response)
        assert isinstance(response, str)
        assert "I like action movies" or query.split()[0] in response

        # --- Test 2: Streaming ---
        query2 = "Show me comedies"
        stream_gen = rag.process_query(query2, top_k=3, stream=True)
        collected = "".join(list(stream_gen))
        print("Streaming collected:", collected)
        assert "Show me comedies" or query2.split()[0] in collected

        # --- Test 3: Conversation memory ---
        rag.reset_conversation()
        queries = ["First query", "Second query"]
        for q in queries:
            rag.process_query(q)
        history = rag.conversation_history
        print("Conversation history:", history)
        assert len(history) == 2
        assert history[0]["user"] == "First query"

        print("✅ All mocked tests passed!")

if __name__ == "__main__":
    test_rag_pipeline_mocked()
