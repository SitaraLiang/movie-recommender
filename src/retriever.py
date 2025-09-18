"""
Movie Retrieval System

Goal: Retrieve similar movies based on user queries using FAISS index and embeddings.
This module handles the retrieval component of the RAG pipeline.

Responsibilities:
- Load FAISS index and embeddings
- Process user queries into embeddings
- Perform similarity search
- Return relevant movie chunks with metadata
"""

import pandas as pd
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class MovieRetriever:
    """
    Movie retrieval system using FAISS index and sentence transformers.
    """

    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initialize the movie retriever.
        
        Args:
            processed_data_dir: Path to directory containing processed data
        """
        # Store the data directory path
        self.processed_data_dir = Path(processed_data_dir)
        
        # Load embedding model
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FAISS index
        logger.info("Loading FAISS index...")
        faiss_index_path = self.processed_data_dir / "movie_embeddings.index"
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        
        # Load movie metadata
        logger.info("Loading movie metadata...")
        movie_metadata_path = self.processed_data_dir / "movie_metadata.csv"
        self.movie_metadata = pd.read_csv(movie_metadata_path)
        
        # Load chunk metadata
        logger.info("Loading chunk metadata...")
        chunk_metadata_path = self.processed_data_dir / "chunk_metadata.csv"
        self.chunk_metadata = pd.read_csv(chunk_metadata_path)
        
        # Load model info
        logger.info("Loading model info...")
        model_info_path = self.processed_data_dir / "model_info.pkl"
        with open(model_info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        logger.info("Retriever initialization completed successfully!")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert a text query into an embedding vector.
        
        Args:
            query: User's text query (e.g., "I like action movies")
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.embedding_model.encode(query, show_progress_bar=False)
        return embedding

    def search_similar_movies(self, query_embedding: np.ndarray, 
                            top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar movies using FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar movies to retrieve
            
        Returns:
            Tuple of (distances, indices) from FAISS search
        """
        # Reshape query embedding to 2D (FAISS expects batch format)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Perform FAISS search
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        return distances[0], indices[0]  # Return 1D arrays

    def get_movie_chunks_by_indices(self, indices: np.ndarray) -> List[Dict]:
        """
        Retrieve movie chunk information by FAISS indices (global_id).
        
        Args:
            indices: Array of FAISS indices (embedding IDs)
            
        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []
        for idx in indices:
            match = self.chunk_metadata[self.chunk_metadata['global_id'] == idx]
            if not match.empty:
                chunk_info = match.iloc[0].to_dict()
                chunks.append(chunk_info)
            else:
                logger.warning(f"No chunk found for global_id {idx}")
        return chunks


    def get_movie_details(self, movie_ids: List[int]) -> pd.DataFrame:
        """
        Get detailed movie information by movie IDs.
        
        Args:
            movie_ids: List of movie IDs
            
        Returns:
            DataFrame with movie details
        """
        # Filter movie_metadata by movie_ids
        movie_details = self.movie_metadata[self.movie_metadata['movieId'].isin(movie_ids)]
        return movie_details

    def format_retrieved_context(self, chunks: List[Dict], 
                               movie_details: pd.DataFrame) -> List[str]:
        """
        Format retrieved chunks and movie details into context strings.
        
        Args:
            chunks: List of chunk dictionaries
            movie_details: DataFrame with movie information
            
        Returns:
            List of formatted context strings for RAG
        """
        context_strings = []
        
        # Create a mapping from movie_id to movie details
        movie_details_dict = movie_details.set_index('movieId').to_dict('index')
        
        for chunk in chunks:
            movie_id = chunk['movie_id']
            
            if movie_id in movie_details_dict:
                movie_info = movie_details_dict[movie_id]
                
                # Format the context string
                context_parts = []
                
                # Add title
                if 'clean_title' in movie_info:
                    context_parts.append(f"Title: {movie_info['clean_title']}")
                
                # Add genres
                if 'clean_genres' in movie_info and movie_info['clean_genres']:
                    genres = ', '.join(eval(movie_info['clean_genres'])) if isinstance(movie_info['clean_genres'], str) else movie_info['clean_genres']
                    context_parts.append(f"Genres: {genres}")
                
                # Add year
                if 'year' in movie_info and movie_info['year'] > 0:
                    context_parts.append(f"Year: {movie_info['year']}")
                
                # Add rating
                if 'avg_rating' in movie_info and movie_info['avg_rating'] > 0:
                    context_parts.append(f"Rating: {movie_info['avg_rating']:.1f}/5.0")

                context_string = " , ".join(context_parts)
                context_strings.append(context_string)
        
        return context_strings

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Main retrieval method - complete pipeline from query to context.
        
        Args:
            query: User's text query
            top_k: Number of movies to retrieve
            
        Returns:
            List of formatted context strings for RAG
        """
        # 1. Encode the query into embedding
        query_embedding = self.encode_query(query)
        
        # 2. Search for similar movies using FAISS
        distances, indices = self.search_similar_movies(query_embedding, top_k)
        
        # 3. Get movie chunks by FAISS indices
        chunks = self.get_movie_chunks_by_indices(indices)
        
        # 4. Extract movie IDs from chunks
        movie_ids = [chunk['movie_id'] for chunk in chunks]
        
        # 5. Get movie details
        movie_details = self.get_movie_details(movie_ids)
        
        # 6. Format retrieved context
        context_strings = self.format_retrieved_context(chunks, movie_details)
        
        return context_strings

    def retrieve_by_genre(self, genre: str, top_k: int = 10) -> List[str]:
        """
        Retrieve movies by specific genre.
        
        Args:
            genre: Genre name (e.g., "Action", "Comedy")
            top_k: Number of movies to retrieve
            
        Returns:
            List of formatted context strings
        """
        # Filter movies by genre
        genre_movies = self.movie_metadata[
            self.movie_metadata['clean_genres'].str.contains(genre, case=False, na=False)
        ]
        
        # Sort by rating and get top_k
        top_movies = genre_movies.sort_values('avg_rating', ascending=False).head(top_k)
        
        # Format the results
        context_strings = []
        for _, movie in top_movies.iterrows():
            context_parts = []
            
            if 'clean_title' in movie:
                context_parts.append(f"Title: {movie['clean_title']}")
            
            if 'clean_genres' in movie and movie['clean_genres']:
                genres = ', '.join(eval(movie['clean_genres'])) if isinstance(movie['clean_genres'], str) else movie['clean_genres']
                context_parts.append(f"Genres: {genres}")
            
            if 'year' in movie and movie['year'] > 0:
                context_parts.append(f"Year: {movie['year']}")
            
            if 'avg_rating' in movie and movie['avg_rating'] > 0:
                context_parts.append(f"Rating: {movie['avg_rating']:.1f}/5.0")
            
            context_string = " | ".join(context_parts)
            context_strings.append(context_string)
        
        return context_strings

    def retrieve_by_year_range(self, start_year: int, end_year: int, 
                             top_k: int = 10) -> List[str]:
        """
        Retrieve movies within a specific year range.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            top_k: Number of movies to retrieve
            
        Returns:
            List of formatted context strings
        """
        # Filter movies by year range
        year_movies = self.movie_metadata[
            (self.movie_metadata['year'] >= start_year) & 
            (self.movie_metadata['year'] <= end_year)
        ]
        
        # Sort by rating and get top_k
        top_movies = year_movies.sort_values('avg_rating', ascending=False).head(top_k)
        
        # Format the results (reuse the same formatting logic)
        context_strings = []
        for _, movie in top_movies.iterrows():
            context_parts = []
            
            if 'clean_title' in movie:
                context_parts.append(f"Title: {movie['clean_title']}")
            
            if 'clean_genres' in movie and movie['clean_genres']:
                genres = ', '.join(eval(movie['clean_genres'])) if isinstance(movie['clean_genres'], str) else movie['clean_genres']
                context_parts.append(f"Genres: {genres}")
            
            if 'year' in movie and movie['year'] > 0:
                context_parts.append(f"Year: {movie['year']}")
            
            if 'avg_rating' in movie and movie['avg_rating'] > 0:
                context_parts.append(f"Rating: {movie['avg_rating']:.1f}/5.0")
            
            context_string = " | ".join(context_parts)
            context_strings.append(context_string)
        
        return context_strings

    def retrieve_by_rating(self, min_rating: float, top_k: int = 10) -> List[str]:
        """
        Retrieve movies with minimum average rating.
        
        Args:
            min_rating: Minimum average rating (e.g., 4.0)
            top_k: Number of movies to retrieve
            
        Returns:
            List of formatted context strings
        """
        # Filter movies by minimum rating
        high_rated_movies = self.movie_metadata[
            self.movie_metadata['avg_rating'] >= min_rating
        ]
        
        # Sort by rating and get top_k
        top_movies = high_rated_movies.sort_values('avg_rating', ascending=False).head(top_k)
        
        # Format the results (reuse the same formatting logic)
        context_strings = []
        for _, movie in top_movies.iterrows():
            context_parts = []
            
            if 'clean_title' in movie:
                context_parts.append(f"Title: {movie['clean_title']}")
            
            if 'clean_genres' in movie and movie['clean_genres']:
                genres = ', '.join(eval(movie['clean_genres'])) if isinstance(movie['clean_genres'], str) else movie['clean_genres']
                context_parts.append(f"Genres: {genres}")
            
            if 'year' in movie and movie['year'] > 0:
                context_parts.append(f"Year: {movie['year']}")
            
            if 'avg_rating' in movie and movie['avg_rating'] > 0:
                context_parts.append(f"Rating: {movie['avg_rating']:.1f}/5.0")
            
            context_string = " | ".join(context_parts)
            context_strings.append(context_string)
        
        return context_strings

    def get_retriever_stats(self) -> Dict:
        """
        Get statistics about the retriever and dataset.
        
        Returns:
            Dictionary with retriever statistics
        """
        stats = {
            'total_movies': len(self.movie_metadata),
            'total_chunks': len(self.chunk_metadata),
            'faiss_index_size': self.faiss_index.ntotal,
            'embedding_dimension': self.faiss_index.d,
            'model_name': self.model_info.get('model_name', 'Unknown'),
            'year_statistics': self.model_info.get('year_statistics', {}),
            'total_embeddings': self.model_info.get('total_embeddings', 0)
        }
        return stats

