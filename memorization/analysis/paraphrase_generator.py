import torch
from typing import List, Dict, Any, Optional
import logging

from memorization.analysis.paraphrase_generation_utils import (
    get_prompts,
    validate_paraphrase_format,
    clean_paraphrase,
    generate_single_paraphrase,
    generate_with_beam_search,
    generate_with_multiple_prompts
)

logger = logging.getLogger(__name__)


class ParaphraseGenerator:
    """Generate paraphrases for text using various strategies

    Four generation modes:
    1. 'greedy_Nprompts': Greedy decoding with N prompts (do_sample=False, num_beams=1)
    2. 'beam_single_prompt': Beam search with 1 prompt, return K outputs (do_sample=False, num_beams=K)
    3. 'beam_Nprompts': Beam search with N prompts (do_sample=False, num_beams=K)
    4. 'nucleus_Nprompts': Nucleus sampling with N prompts (do_sample=True, top_p=0.9, temperature=0.7)

    N is determined by num_paraphrases parameter (max 5 prompts available)
    """

    def __init__(self, config):
        """Initialize ParaphraseGenerator

        Args:
            config: Configuration object with paraphrasing settings
        """
        self.config = config

        # Extract paraphrasing settings
        self.num_paraphrases = getattr(config.analysis, 'num_paraphrases', 5)
        self.num_beams = getattr(config.analysis, 'num_beams', 4)
        self.temperature = getattr(config.analysis, 'paraphrase_temperature', 0.5)
        self.top_p = getattr(config.analysis, 'top_p', 0.9)
        self.top_k = getattr(config.analysis, 'top_k', 40)
        self.repetition_penalty = getattr(config.analysis, 'repetition_penalty', 1.1)
        self.use_soft_beam_sampling = getattr(config.analysis, 'use_soft_beam_sampling', False)

        # Generation mode: 'greedy_Nprompts', 'beam_single_prompt', 'beam_Nprompts', 'nucleus_Nprompts'
        self.generation_mode = getattr(config.analysis, 'generation_mode', 'beam_Nprompts')

        logger.info(f"ParaphraseGenerator initialized: mode={self.generation_mode}, "
                   f"num_paraphrases={self.num_paraphrases}, num_beams={self.num_beams}, "
                   f"temperature={self.temperature}, top_p={self.top_p}")

    def _generate_paraphrases(self, text: str, model, tokenizer, input_type: str = 'text', question: str = "") -> List[Dict[str, Any]]:
        """Generate paraphrases for a single text

        Args:
            text: Input text to paraphrase
            model: Language model for generation
            tokenizer: Tokenizer
            input_type: Kept for backward compatibility (ignored)
            question: Kept for backward compatibility (ignored)

        Returns:
            List of paraphrase dictionaries with 'text', 'score', and 'method' fields
        """
        # Select generation strategy based on mode
        if self.generation_mode == 'greedy_Nprompts':
            # Use N prompts (determined by num_paraphrases) with greedy decoding
            prompts = get_prompts(text, num_prompts=self.num_paraphrases)
            return generate_with_multiple_prompts(
                text, model, tokenizer, prompts, use_beam=False,
                num_paraphrases=self.num_paraphrases, num_beams=self.num_beams
            )

        elif self.generation_mode == 'beam_single_prompt':
            # Use 1 prompt with beam search to generate K outputs
            prompts = get_prompts(text, num_prompts=1)
            return generate_with_beam_search(
                text, model, tokenizer, prompts[0], return_multiple=True,
                num_paraphrases=self.num_paraphrases, num_beams=self.num_beams,
                temperature=self.temperature, top_p=self.top_p,
                use_soft_sampling=self.use_soft_beam_sampling
            )

        elif self.generation_mode == 'beam_Nprompts':
            # Use N prompts (determined by num_paraphrases) with beam search
            prompts = get_prompts(text, num_prompts=self.num_paraphrases)
            return generate_with_multiple_prompts(
                text, model, tokenizer, prompts, use_beam=True,
                num_paraphrases=self.num_paraphrases, num_beams=self.num_beams
            )

        elif self.generation_mode == 'nucleus_Nprompts':
            # Use N prompts (determined by num_paraphrases) with nucleus sampling
            prompts = get_prompts(text, num_prompts=self.num_paraphrases)
            return generate_with_multiple_prompts(
                text, model, tokenizer, prompts, use_beam=False,
                num_paraphrases=self.num_paraphrases, num_beams=self.num_beams,
                temperature=self.temperature, top_p=self.top_p, use_sampling=True
            )

        else:
            logger.error(f"Unknown generation mode: {self.generation_mode}")
            return [{'text': text, 'score': 1.0, 'method': 'original'}]

    def generate_paraphrases_batch(self, texts: List[str], model, tokenizer) -> List[List[Dict[str, Any]]]:
        """Generate paraphrases for a batch of texts in parallel (GPU acceleration)

        Args:
            texts: List of input texts to paraphrase
            model: Language model for generation
            tokenizer: Tokenizer

        Returns:
            List of paraphrase lists, one per input text
        """
        if not texts:
            return []

        # Only batch generation supported for beam_single_prompt mode
        # For other modes, fall back to sequential processing
        if self.generation_mode != 'beam_single_prompt':
            logger.warning(f"Batch generation not optimized for {self.generation_mode}, using sequential")
            return [self._generate_paraphrases(text, model, tokenizer) for text in texts]

        try:
            # Prepare prompts for all texts in batch
            batch_prompts = []
            for text in texts:
                prompts = get_prompts(text, num_prompts=1)
                batch_prompts.append(prompts[0])

            # Tokenize all prompts together with padding
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Decoder-only 모델은 left-padding 필수 (배치 생성 시)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,  # Pad to longest sequence in batch
                truncation=True,
                max_length=2048
            ).to(model.device)

            # 원래 설정으로 복원
            tokenizer.padding_side = original_padding_side

            # Calculate max_new_tokens based on longest text in batch
            max_original_tokens = max(len(tokenizer.tokenize(text)) for text in texts)
            max_new_tokens = int(max_original_tokens * 1.1) + 15
            min_new_tokens = max(10, int(max_original_tokens * 0.9))

            # Generate for entire batch at once
            # Ensure num_beams >= num_return_sequences (required by HuggingFace)
            effective_num_beams = max(self.num_beams, self.num_paraphrases)

            with torch.no_grad():
                if self.use_soft_beam_sampling:
                    # Controlled sampling with beam search
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=effective_num_beams,
                        num_return_sequences=self.num_paraphrases,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        early_stopping=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    # Deterministic beam search
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=effective_num_beams,
                        num_return_sequences=self.num_paraphrases,
                        do_sample=False,
                        early_stopping=False,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=self.repetition_penalty
                    )

            # Process outputs: outputs shape is (batch_size * num_return_sequences, seq_len)
            batch_size = len(texts)
            all_paraphrases = []

            for i in range(batch_size):
                # Get all sequences for this input
                start_idx = i * self.num_paraphrases
                end_idx = start_idx + self.num_paraphrases
                text_outputs = outputs[start_idx:end_idx]

                # Get input length for this specific text (accounting for padding)
                input_length = (inputs.attention_mask[i] == 1).sum().item()

                paraphrases = []
                for j, output in enumerate(text_outputs):
                    # Decode generated tokens (skip input prompt)
                    generated_tokens = output[input_length:]
                    paraphrase_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    # Import clean_paraphrase and validate_paraphrase_format from utils
                    from memorization.analysis.paraphrase_generation_utils import clean_paraphrase, validate_paraphrase_format

                    paraphrase_text = clean_paraphrase(paraphrase_text)

                    if paraphrase_text and len(paraphrase_text.split()) >= 3:
                        if validate_paraphrase_format(paraphrase_text):
                            paraphrases.append({
                                'text': paraphrase_text,
                                'score': 1.0 / (j + 1),
                                'method': 'batch_beam',
                                'beam_idx': j
                            })

                # Fallback if no valid paraphrases generated
                if not paraphrases:
                    paraphrases = [{'text': texts[i], 'score': 1.0, 'method': 'original'}] * self.num_paraphrases

                all_paraphrases.append(paraphrases)

            logger.info(f"✅ Batch generated paraphrases for {batch_size} texts")
            return all_paraphrases

        except Exception as e:
            logger.error(f"Batch generation failed: {e}, falling back to sequential")
            return [self._generate_paraphrases(text, model, tokenizer) for text in texts]
