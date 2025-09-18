"""
Movie Recommendation Generator

Goal: Generate movie recommendations using LLaMA-2-7B-chat-hf model with RAG context.
This module handles the generation component of the RAG pipeline.

Responsibilities:
- Load and manage LLaMA-2-7B-chat-hf (or Azure-hosted) model
- Format retrieved context into prompts
- Generate natural language recommendations
- Provide explanations for recommendations
- Optional: support streaming tokens for chat UI
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.models.prompt_depth_anything.modular_prompt_depth_anything import PromptDepthAnythingFeatureFusionStage

class MovieGenerator:
    """
    LLM-backed recommendation generator that turns retrieved context + user query
    into natural-language recommendations and explanations.
    """

    def __init__(self,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        #model_name="sshleifer/tiny-gpt2", # Use a small model for debugging
        device: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 256
    ) -> None:
        """
        Configure generator parameters. Does not necessarily load weights yet.

        Args:
            model_name: HF model id (chat-tuned) or Azure model deployment name
            device: Preferred device (e.g., "cpu", "cuda", "mps")
            temperature: Sampling temperature
            top_p: Nucleus sampling probability (Lower p → more deterministic output; higher p → more diverse output.)
            max_new_tokens: Max tokens to generate per response
            use_azure: Whether to call Azure-hosted LLM instead of local/HF pipeline
            azure_endpoint: Azure endpoint URL if use_azure is True
            azure_api_key: Azure API key/credential
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.system_prompt = "You are a helpful movie recommender."

        # Load Hugging Face model + tokenizer
        self.backend = "huggingface" 
        # automatically download and cache locally the right tokenizer for the given model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device is None else None # place the model across available hardware automatically
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            #device=0 if self.device == "cuda" else -1, # -1: CPU or MPS
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or override the system prompt to guide the assistant persona.
        """
        self.system_prompt = prompt

    def build_context_block(self, contexts: List[str]) -> str:
        """
        Build a formatted context block from retrieved movie snippets for RAG.

        Args:
            contexts: Ranked list of context strings from the retriever
            max_items: Limit number of context items included

        Returns:
            A single formatted string to insert into the final prompt.
        """
        if not contexts:
            return ""

        # Format them as a numbered list
        context_lines = [f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)]

        # Join lines into a single block
        context_block = "Context:\n" + "\n".join(context_lines)

        return context_block

    def build_prompt(self, user_query: str, context_block: str) -> str:
        """
        Construct the final prompt for the chat model given user query and context.

        Args:
            user_query: The user message (e.g., "I loved The Matrix...")
            context_block: Formatted context from build_context_block

        Returns:
            Full prompt string ready for generation.
        """
        prompt = f"System: {self.system_prompt}\n"
        if context_block:
            prompt += f"{context_block}\n"
        prompt += f"User: {user_query}\nAssistant:"
        return prompt

    def generate_text(self, prompt: str) -> str:
        """
        Non-streaming, generate a response from the model for a single prompt

        Args:
            prompt: Prepared prompt including system + context + user parts

        Returns:
            The generated text.
        """
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return outputs[0]["generated_text"][len(prompt):].strip()


    def generate_stream(self, prompt: str):
        """
        Streaming, yields tokens one by one

        Args:
            prompt: Prepared prompt including system + context + user parts
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            return_dict_in_generate=True,
            output_scores=True
        )
        for token_id in generated_ids.sequences[0][input_ids.shape[-1]:]:
            yield self.tokenizer.decode(token_id, skip_special_tokens=True)

    def generate_with_context(self, user_query: str, contexts: List[str], stream: bool = False) -> str:
        """
        Convenience method: format contexts + build prompt + generate.

        Args:
            user_query: The user question or instruction
            contexts: Retrieved context snippets from the retriever
            stream: False → wait for the full response (simpler, standard)
                    True → get tokens incrementally (useful for UI/UX like live typing)

        Returns:
            Generated answer string.
        """
        # 1. Build context block
        context_block = self.build_context_block(contexts)

        # 2. Build prompt
        prompt = self.build_prompt(user_query, context_block)

        # 3. Generate response
        if stream:
            return self.generate_stream(prompt)
        else:
            return self.generate_text(prompt)

    def extract_recommendations(self, text: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """
        Parse the LLM output to extract structured recommendations.
        The exact heuristic/regex is up to the implementer.

        Returns:
            A list of dicts like {"title": str, "reason": str, "year": Optional[int]}.
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for budgeting/truncation.
        """
        # we count only the raw text tokens, ignoring special tokens like <s> or </s>.
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

