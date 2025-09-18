"""
Data Preprocessing

This script handles:
1. Chunking synopses/metadata into retrievable text
2. Generating embeddings with sentence-transformers/all-MiniLM-L6-v2
3. Storing embeddings in FAISS index
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovieDataPreprocessor:
    """Handles preprocessing of movie data for recommendation system."""
    
    def __init__(self, data_dir= "data/ml-latest-small", output_dir= "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True) # create the output directory if it doesn't exist
        
        # Initialize embedding model
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Data storage
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.links_df = None
        
    def load_data(self) -> None:
        """Load all raw data files."""
        logger.info("Loading raw data files...")
        
        self.movies_df = pd.read_csv(self.data_dir / "movies.csv")
        self.ratings_df = pd.read_csv(self.data_dir / "ratings.csv")
        self.tags_df = pd.read_csv(self.data_dir / "tags.csv")
        self.links_df = pd.read_csv(self.data_dir / "links.csv")
        
        logger.info(f"Loaded {len(self.movies_df)} movies, {len(self.ratings_df)} ratings, "
                   f"{len(self.tags_df)} tags, {len(self.links_df)} links")
    
    def clean_title(self, title) -> str:
        """Clean and normalize movie title."""
        # if the title is not a string, return an empty string
        if pd.isna(title):
            return ""
        
        # Remove year in parentheses
        # \s* matches any whitespace characters (spaces, tabs, newlines) zero or more times
        # \( matches the literal character "("
        # \d{4} matches exactly four digits
        # \) matches the literal character ")"
        # \s* matches any whitespace characters (spaces, tabs, newlines) zero or more times
        # $ asserts the position at the end of the string
        title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
        
        # Remove special characters but keep spaces and basic punctuation
        # [^\w\s\-&] matches any character that is not a word character (letter, digit, underscore), space, hyphen, or ampersand
        # ^ asserts the position at the beginning of the string
        # $ asserts the position at the end of the string
        title = re.sub(r'[^\w\s\-&]', '', title)
        
        # Normalize whitespace
        # \s+ matches one or more whitespace characters (spaces, tabs, newlines)
        # .strip() removes any leading or trailing whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def extract_year(self, title) -> int:
        """
        Extract year from movie title.
        Returns the year as an integer, or 0 if no year is found.
        """
        if pd.isna(title):
            return 0
        
        # Look for year in parentheses at the end of the title
        # Pattern: (YYYY) at the end of the string
        # If the title contains a year in parentheses at the end, year_match will hold a match object.
        # That object has the full match and any capture groups.
        year_match = re.search(r'\((\d{4})\)\s*$', str(title))
        
        if year_match:
            # year_match.group(0) → the entire match
            # year_match.group(1) → the first capturing group
            year = int(year_match.group(1))
            # Validate year range (reasonable movie years)
            if 1888 <= year <= 2025: 
                return year
        
        return 0
    
    def clean_genres(self, genres) -> List[str]:
        """Clean and normalize genres."""
        if pd.isna(genres) or genres == "(no genres listed)":
            return []
        
        # Split by pipe and clean each genre
        genre_list = [genre.strip() for genre in genres.split('|')]
        
        # Remove empty genres and normalize
        genre_list = [genre for genre in genre_list if genre and genre != "(no genres listed)"]
        
        return genre_list
    
    def clean_tags(self, tags) -> List[str]:
        """
        Clean and normalize tags.
        Each tag is typically a single word or short phrase
        """
        if not tags:
            return []
        
        cleaned_tags = []
        for tag in tags:
            if pd.isna(tag):
                continue
                
            # Convert to lowercase and remove special characters
            tag = re.sub(r'[^\w\s]', '', str(tag).lower())
            
            # Remove extra whitespace
            tag = re.sub(r'\s+', ' ', tag).strip()
            
            # Skip empty tags and very short tags
            if tag and len(tag) > 1:
                cleaned_tags.append(tag)
        
        return list(set(cleaned_tags))  # set(): Remove duplicates, returns an unordered collection {}
    
    def create_movie_metadata(self) -> pd.DataFrame:
        """Create comprehensive movie metadata by combining all data sources."""
        logger.info("Creating comprehensive movie metadata...")
        
        # Start with movies data
        # makes a copy and then applies cleaning functions
        metadata = self.movies_df.copy()
        
        # Clean titles and genres, extract year
        metadata['clean_title'] = metadata['title'].apply(self.clean_title)
        metadata['clean_genres'] = metadata['genres'].apply(self.clean_genres)
        metadata['year'] = metadata['title'].apply(self.extract_year)
                
        # Drop the raw, unclean column
        metadata = metadata.drop('title', axis=1)
        metadata = metadata.drop('genres', axis=1)
        
        # Add rating statistics
        rating_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count'],
            'userId': 'nunique'
        }).round(2)
        
        # Rename the columns
        rating_stats.columns = ['avg_rating', 'rating_std', 'rating_count', 'unique_users']
        rating_stats = rating_stats.reset_index()
        
        # After groupby, movieId becomes the index.
        # .reset_index() turns it back into a normal column so it can be merged later.
        # how='left' : movies without ratings get NaN.
        metadata = metadata.merge(rating_stats, on='movieId', how='left')
        
        # Add tags
        # ['tag'].apply(list) : collect tags into a Python list for each movie.
        # .reset_index() : turn movieId back into a column.
        movie_tags = self.tags_df.groupby('movieId')['tag'].apply(list).reset_index()
        movie_tags['clean_tags'] = movie_tags['tag'].apply(self.clean_tags)
        # Drop the raw, unclean tag column → only keep movieId and clean_tags.
        movie_tags = movie_tags.drop('tag', axis=1)
    
        # how='left' : movies without tags get NaN.
        metadata = metadata.merge(movie_tags, on='movieId', how='left')
        # Replace NaN with '' and ensure everything is a list.
        # So every movie ends up with clean_tags as a list (possibly empty).
        metadata['clean_tags'] = metadata['clean_tags'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
        
        # Add external IDs
        metadata = metadata.merge(self.links_df, on='movieId', how='left')
        
        # Fill missing values
        # If a movie has no ratings, the stats are NaN.
        # Fill them with 0 -> prevents errors later when doing calculations or displaying results.
        metadata['avg_rating'] = metadata['avg_rating'].fillna(0)
        metadata['rating_std'] = metadata['rating_std'].fillna(0)
        metadata['rating_count'] = metadata['rating_count'].fillna(0)
        metadata['unique_users'] = metadata['unique_users'].fillna(0)
        
        return metadata
    
    def create_text_chunks(self, row: pd.Series) -> List[str]:
        """Create text chunks for a movie from its metadata."""
        chunks = []
        
        # Title chunk
        if row['clean_title']:
            chunks.append(f"Title: {row['clean_title']}")
        
        # Genre chunk
        if row['clean_genres']:
            genre_text = ", ".join(row['clean_genres'])
            chunks.append(f"Genres: {genre_text}")
        
        # Rating chunk
        if row['avg_rating'] > 0:
            rating_info = f"Average rating: {row['avg_rating']:.1f}/5.0"
            if row['rating_count'] > 0:
                rating_info += f" from {int(row['rating_count'])} ratings"
            chunks.append(rating_info)
        
        # Tags chunk
        if row['clean_tags']:
            # Limit tags to avoid overly long chunks
            tags_text = ", ".join(row['clean_tags'])
            chunks.append(f"Tags: {tags_text}")
        
        # Year chunk (using extracted year)
        if row['year'] > 0:
            chunks.append(f"Release year: {row['year']}")
        
        return chunks
    
    def generate_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        if not text_chunks:
            # Return zero vector if no chunks
            return np.zeros((1, 384))  # all-MiniLM-L6-v2 has 384 dimensions
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(text_chunks, show_progress_bar=False)
        return embeddings
    
    def create_faiss_index(self, embeddings_list: List[np.ndarray]) -> faiss.Index:
        """Create FAISS index from embeddings."""
        logger.info("Creating FAISS index...")
        
        try:
            # Combine all embeddings
            logger.info(f"Combining {len(embeddings_list)} embedding arrays...")
            # Combines multiple arrays of embeddings into one big 2D array. 
            # Example: if you had 3 arrays of shape (100, 384), (200, 384), (50, 384), 
            # the result is (100+200+50, 384).
            all_embeddings = np.vstack(embeddings_list)
            logger.info(f"Combined embeddings shape: {all_embeddings.shape}")
            
            # Create FAISS index
            dimension = all_embeddings.shape[1] # 384
            logger.info(f"Creating FAISS index with dimension {dimension}")
            
            # Use IndexFlatL2 instead of IndexFlatIP to avoid potential issues
            # L2 (Euclidean) distance for similarity search.
            # “flat” = brute-force exact search, no approximations.
            index = faiss.IndexFlatL2(dimension)
            
            # Convert to float32 and add in batches to avoid memory issues
            batch_size = 1000
            total_vectors = all_embeddings.shape[0] # = sum(len(emb) for emb in embeddings_list)
            
            logger.info(f"Adding {total_vectors} vectors in batches of {batch_size}")
            
            for i in range(0, total_vectors, batch_size):
                end_idx = min(i + batch_size, total_vectors)
                # float32: FAISS requires
                batch = all_embeddings[i:end_idx].astype('float32')
                # Add batch to the FAISS index.
                index.add(batch)
                
                if (i // batch_size) % 10 == 0:
                    logger.info(f"Added {end_idx}/{total_vectors} vectors")
            
            logger.info(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}")
            
            return index
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            # Fallback: create a simple index with just the first few embeddings
            logger.info("Creating fallback FAISS index...")
            # If there is at least one set of embeddings
            if embeddings_list:
                sample_embeddings = embeddings_list[0][:100]  # Take first 100 embeddings
                dimension = sample_embeddings.shape[1] # 384
                index = faiss.IndexFlatL2(dimension)
                index.add(sample_embeddings.astype('float32'))
                logger.info(f"Created fallback FAISS index with {index.ntotal} vectors")
                return index
            else:
                raise e
    
    def process_data(self) -> None:
        """Main processing pipeline."""
        logger.info("Starting data preprocessing pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Create metadata
            # a metadata DataFrame with one row per movie, ready for chunking
            metadata = self.create_movie_metadata()
            
            # Create text chunks and embeddings
            logger.info("Creating text chunks and generating embeddings...")
            embeddings_list = []
            chunk_metadata = []
            
            # Process in batches to manage memory
            batch_size = 1000 # avoid memory overload
            total_movies = len(metadata)
            global_id = 0
            
            for batch_start in range(0, total_movies, batch_size):
                batch_end = min(batch_start + batch_size, total_movies)
                batch_metadata = metadata.iloc[batch_start:batch_end]
                
                logger.info(f"Processing movies {batch_start+1}-{batch_end} of {total_movies}")
                
                # Process each movie in the batch
                for idx, row in batch_metadata.iterrows():
                    try:
                        # First splits movie info into manageable pieces 
                        chunks = self.create_text_chunks(row) 
                        # Then converts each text chunk into a vector (NumPy array)
                        embeddings = self.generate_embeddings(chunks)
                        # Appends embeddings to list
                        embeddings_list.append(embeddings)
                        
                        # Store metadata for each chunk
                        # This helps later when doing search or retrieving results.
                        for i, chunk in enumerate(chunks):
                            chunk_metadata.append({
                                'movie_id': row['movieId'],
                                'chunk_id': i,
                                'chunk_text': chunk,
                                'embedding_id': len(embeddings_list) - 1, # points to the movie in embeddings_list.
                                'global_id': global_id
                            })
                            global_id += 1
                    # If a single movie fails (bad text, encoding error, etc.), log a warning and skip it.
                    except Exception as e:
                        logger.warning(f"Error processing movie {row['movieId']}: {e}")
                        continue
                
                # Clear memory periodically
                # Prevents memory from growing too large during embedding creation.
                if batch_start % (batch_size * 5) == 0:
                    import gc
                    gc.collect()
            
            logger.info(f"Generated embeddings for {len(embeddings_list)} movies")
            
            # Create FAISS vector index for fast similarity search across all movie embeddings
            # Note: each element in embeddings_list is a NumPy array for one movie.
            # the FAISS method stacks them and adds to the index
            faiss_index = self.create_faiss_index(embeddings_list)
            
            # Save processed data
            # metadata -> full movie info
            # chunk_metadata -> mapping of chunks to movies
            # faiss_index -> search index
            # embeddings_list -> raw embeddings for reuse
            self.save_processed_data(metadata, chunk_metadata, faiss_index, embeddings_list)
            
            logger.info("Data preprocessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def save_processed_data(self, metadata: pd.DataFrame, chunk_metadata: List[Dict], 
                           faiss_index: faiss.Index, embeddings_list: List[np.ndarray]) -> None:
        """Save all processed data."""
        logger.info("Saving processed data...")
        
        # Save metadata
        metadata.to_csv(self.output_dir / "movie_metadata.csv", index=False)
        
        # Save chunk metadata
        chunk_df = pd.DataFrame(chunk_metadata)
        chunk_df.to_csv(self.output_dir / "chunk_metadata.csv", index=False)
        
        # Save FAISS index
        # faiss_index is a specialized C++ object managed by the FAISS library.
        # FAISS has its own binary format for saving indexes (.index files).
        # This format stores not just embeddings, but also:
        #   - Index structure (IVF, HNSW, Flat, etc.)
        #   - Quantization/compression info
        #   - Search parameters
        faiss.write_index(faiss_index, str(self.output_dir / "movie_embeddings.index"))
        
        # Save embeddings as numpy arrays
        # pickle (.pkl): serializes any Python object into bytes and lets us load it back later.
        with open(self.output_dir / "embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings_list, f)
        
        # Save embedding model info
        year_stats = {
            'min_year': int(metadata['year'].min()) if metadata['year'].min() > 0 else 0,
            'max_year': int(metadata['year'].max()) if metadata['year'].max() > 0 else 0,
            'movies_with_year': int((metadata['year'] > 0).sum()),
            'movies_without_year': int((metadata['year'] == 0).sum())
        }
        
        model_info = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dimension': 384, 
            'total_movies': len(metadata),
            'total_chunks': len(chunk_metadata),
            'total_embeddings': sum(len(emb) for emb in embeddings_list),
            'year_statistics': year_stats
        }
        
        with open(self.output_dir / "model_info.pkl", 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Saved processed data to {self.output_dir}")
        logger.info(f"Total movies: {len(metadata)}")
        logger.info(f"Total text chunks: {len(chunk_metadata)}")
        logger.info(f"Total embeddings: {sum(len(emb) for emb in embeddings_list)}")
        logger.info(f"Year range: {year_stats['min_year']}-{year_stats['max_year']}")
        logger.info(f"Movies with year info: {year_stats['movies_with_year']}")
        logger.info(f"Movies without year info: {year_stats['movies_without_year']}")
