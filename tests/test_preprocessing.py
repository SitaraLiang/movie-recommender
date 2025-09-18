"""
Test script for the movie preprocessing pipeline
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import MovieDataPreprocessor

def test_preprocessing():
    """Test the preprocessing pipeline with a small sample."""
    print("Testing movie preprocessing pipeline...")
    
    try:
        # Initialize preprocessor
        preprocessor = MovieDataPreprocessor()
        
        # Load data
        preprocessor.load_data()
        
        # Test with first 10 movies
        print(f"Testing with first 10 movies...")
        
        # Create metadata for sample
        metadata = preprocessor.create_movie_metadata()
        sample_metadata = metadata.head(10)
        
        print("Sample metadata created:")
        print(sample_metadata[['movieId', 'clean_title', 'clean_genres', 'avg_rating', 'clean_tags', 'year']].head())
        
        # Test text chunking
        print("\nTesting text chunking...")
        for idx, row in sample_metadata.iterrows():
            chunks = preprocessor.create_text_chunks(row)
            print(f"Movie {row['movieId']}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i}: {chunk}")
            print()
        
        # Test embedding generation
        print("Testing embedding generation...")
        # .iloc[0] -> Series acts like a dictionary of column-value pairs for that row.
        sample_chunks = preprocessor.create_text_chunks(sample_metadata.iloc[0])
        embeddings = preprocessor.generate_embeddings(sample_chunks) 
        print(f"Generated embeddings shape: {embeddings.shape}") # (5, 384)

        # Test FAISS index creation
        print("Testing FAISS index creation...")
        faiss_index = preprocessor.create_faiss_index(embeddings)
        print(f"Created FAISS index with {faiss_index.ntotal} vectors")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessing()
