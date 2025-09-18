"""
Tests for Movie Retriever

Goal: Comprehensive unit tests for the movie retrieval system.
This module contains tests for all retrieval functionality.

Test Coverage:
- Retriever initialization and data loading
- Query embedding generation
- FAISS similarity search
- Movie chunk retrieval
- Context formatting
- Genre-based retrieval
- Year-based retrieval
- Rating-based retrieval
- Statistics generation
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.retriever import MovieRetriever


class TestMovieRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize retriever once for all tests."""
        cls.retriever = MovieRetriever()

    def test_retriever_initialization(self):
        r = self.retriever
        self.assertIsNotNone(r.embedding_model, "Embedding model not loaded")
        self.assertIsNotNone(r.faiss_index, "FAISS index not loaded")
        self.assertIsNotNone(r.movie_metadata, "Movie metadata not loaded")
        self.assertIsNotNone(r.chunk_metadata, "Chunk metadata not loaded")
        self.assertIsNotNone(r.model_info, "Model info not loaded")

        self.assertGreater(len(r.movie_metadata), 0, "No movies loaded")
        self.assertGreater(len(r.chunk_metadata), 0, "No chunks loaded")
        self.assertGreater(r.faiss_index.ntotal, 0, "Empty FAISS index")

    def test_encode_query(self):
        queries = [
            "I like action movies",
            "Comedy films with good ratings",
            "Sci-fi movies from the 90s",
            "Romantic dramas"
        ]
        for q in queries:
            emb = self.retriever.encode_query(q)
            self.assertIsInstance(emb, np.ndarray)
            self.assertEqual(emb.shape, (384,))
            self.assertFalse(np.isnan(emb).any())
            self.assertFalse(np.isinf(emb).any())

    def test_search_similar_movies(self):
        q_emb = self.retriever.encode_query("action movies")
        distances, indices = self.retriever.search_similar_movies(q_emb, top_k=5)
        self.assertEqual(len(distances), 5)
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(idx >= 0 for idx in indices))
        self.assertTrue(all(idx < self.retriever.faiss_index.ntotal for idx in indices))
        self.assertTrue(all(d >= 0 for d in distances))

    def test_get_movie_chunks_by_indices(self):
        indices = np.array([0, 1, 2, 10, 100])
        chunks = self.retriever.get_movie_chunks_by_indices(indices)
        self.assertEqual(len(chunks), len(indices))
        for ch in chunks:
            self.assertIsInstance(ch, dict)
            self.assertIn('movie_id', ch)
            self.assertIn('chunk_text', ch)
            self.assertIsInstance(ch['movie_id'], (int, np.integer))
            self.assertIsInstance(ch['chunk_text'], str)

    def test_get_movie_details(self):
        ids = [1, 2, 3, 100, 200]
        df = self.retriever.get_movie_details(ids)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertLessEqual(len(df), len(ids))
        self.assertIn('movieId', df.columns)
        self.assertIn('clean_title', df.columns)

    def test_format_retrieved_context(self):
        chunks = [
            {'movie_id': 1, 'chunk_text': 'Title: Test Movie'},
            {'movie_id': 2, 'chunk_text': 'Genres: Action, Drama'}
        ]
        ids = [ch['movie_id'] for ch in chunks]
        details = self.retriever.get_movie_details(ids)
        ctx = self.retriever.format_retrieved_context(chunks, details)
        self.assertIsInstance(ctx, list)
        self.assertLessEqual(len(ctx), len(chunks))
        for c in ctx:
            self.assertIsInstance(c, str)
            self.assertGreater(len(c), 0)

    def test_retrieve_by_genre(self):
        for genre in ['Action', 'Comedy', 'Drama']:
            results = self.retriever.retrieve_by_genre(genre, top_k=3)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            for r in results:
                self.assertIsInstance(r, str)
                self.assertGreater(len(r), 0)

    def test_retrieve_by_year_range(self):
        ranges = [(1990, 1999), (2000, 2009), (2010, 2015)]
        for start, end in ranges:
            results = self.retriever.retrieve_by_year_range(start, end, top_k=3)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            for r in results:
                self.assertIsInstance(r, str)
                self.assertGreater(len(r), 0)

    def test_retrieve_by_rating(self):
        for rating in [3.0, 4.0, 4.5]:
            results = self.retriever.retrieve_by_rating(rating, top_k=3)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            for r in results:
                self.assertIsInstance(r, str)
                self.assertGreater(len(r), 0)

    def test_retrieve_main_pipeline(self):
        queries = [
            "I like action movies",
            "Comedy films with good ratings",
            "Sci-fi movies from the 90s"
        ]
        for q in queries:
            results = self.retriever.retrieve(q, top_k=5)
            self.assertIsInstance(results, list)
            for r in results:
                self.assertIsInstance(r, str)
                self.assertGreater(len(r), 0)

    def test_get_retriever_stats(self):
        stats = self.retriever.get_retriever_stats()
        self.assertIsInstance(stats, dict)
        keys = [
            'total_movies', 'total_chunks', 'faiss_index_size',
            'embedding_dimension', 'model_name', 'year_statistics', 'total_embeddings'
        ]
        for k in keys:
            self.assertIn(k, stats)
        self.assertGreater(stats['total_movies'], 0)
        self.assertGreater(stats['total_chunks'], 0)
        self.assertGreater(stats['faiss_index_size'], 0)
        self.assertEqual(stats['embedding_dimension'], 384)
        self.assertEqual(stats['model_name'], 'sentence-transformers/all-MiniLM-L6-v2')


if __name__ == "__main__":
    unittest.main(verbosity=2)
