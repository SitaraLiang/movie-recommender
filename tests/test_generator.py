"""
Tests for Movie Generator

Goal: Unit tests for the movie recommendation generator.
This module contains tests for the generation functionality.

Test Coverage:
- LLaMA model loading
- Prompt formatting
- Response generation
- Context integration
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from src.generator import MovieGenerator

class TestMovieGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize generator once for all tests
        cls.generator = MovieGenerator(
            #model_name="meta-llama/Llama-2-7b-chat-hf",
            model_name="sshleifer/tiny-gpt2", # Use a small model for debugging
            device="cpu",  # use CPU for testing
            max_new_tokens=32
        )

    def test_model_loaded(self):
        """Test that the tokenizer and model are loaded correctly."""
        self.assertIsNotNone(self.generator.model)
        self.assertIsNotNone(self.generator.tokenizer)
        self.assertEqual(self.generator.backend, "huggingface")

    def test_set_system_prompt(self):
        """Test that system prompt can be updated."""
        new_prompt = "You are a sarcastic movie critic."
        self.generator.set_system_prompt(new_prompt)
        self.assertEqual(self.generator.system_prompt, new_prompt)

    def test_build_prompt(self):
        """Test prompt construction with user query and context."""
        contexts = [
            "Title: Inception , Genres: Sci-Fi, Thriller , Year: 2010 , Rating: 4.8/5.0",
            "Title: Arrival , Genres: Sci-Fi, Drama , Year: 2016 , Rating: 4.7/5.0"
        ]
        context_block = self.generator.build_context_block(contexts)
        user_query = "Recommend a mind-bending sci-fi movie."
        prompt = self.generator.build_prompt(user_query, context_block)
        
        self.assertIn("System:", prompt)
        self.assertIn("User: Recommend a mind-bending sci-fi movie.", prompt)
        self.assertIn("Context:", prompt)
        self.assertIn("Title: Inception", prompt)

    def test_generate_with_context(self):
        """Test that generator produces a non-empty response in both streaming and non-streaming modes."""
        contexts = [
            "Title: Inception , Genres: Sci-Fi, Thriller , Year: 2010 , Rating: 4.8/5.0",
            "Title: Arrival , Genres: Sci-Fi, Drama , Year: 2016 , Rating: 4.7/5.0"
        ]
        user_query = "Recommend a mind-bending sci-fi movie."

        # Non-streaming test (should return a string)
        response = self.generator.generate_with_context(user_query, contexts, stream=False)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response.strip()) > 0)
        print("Non-streaming response:", response)

        # Streaming test (should return a generator)
        response_gen = self.generator.generate_with_context(user_query, contexts, stream=True)
        self.assertTrue(hasattr(response_gen, "__iter__"))  # it is a generator

        # Collect all tokens from the generator into a string
        response_text = "".join(list(response_gen))
        self.assertIsInstance(response_text, str)
        self.assertTrue(len(response_text.strip()) > 0)
        print("Streaming response:", response_text)



if __name__ == "__main__":
    unittest.main()

