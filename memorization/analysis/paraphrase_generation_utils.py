import torch
import logging
import os
from typing import List, Dict, Any, Optional
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Get the directory where prompt files are stored
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')


def load_prompt_from_file(filename: str, text: str) -> str:
    """Load a prompt template from file and fill in all placeholders

    Args:
        filename: Name of the prompt file (e.g., 'syntactic_rewrite_1.txt')
        text: Text to insert into the prompt template

    Returns:
        Formatted prompt string with all placeholders replaced
    """
    filepath = os.path.join(PROMPTS_DIR, filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            template = f.read()

        # Define all placeholder values for pile paraphrasing (text-only, no QA)
        INSTRUCTION = 'Keep the output in the same format as the original.'
        EXAMPLE = 'The capital of France is Paris, which is known for the Eiffel Tower.'
        EXAMPLE_OUTPUT = 'Paris is the capital of France, famous for the Eiffel Tower.'
        OUTPUT_START = ''
        QUESTION_CONTEXT = ''
        EXAMPLE_QUESTION_CONTEXT = ''

        # Replace all placeholders
        prompt = template.replace('{QUESTION_CONTEXT}', QUESTION_CONTEXT)
        prompt = prompt.replace('{EXAMPLE_QUESTION_CONTEXT}', EXAMPLE_QUESTION_CONTEXT)
        prompt = prompt.replace('{INSTRUCTION}', INSTRUCTION)
        prompt = prompt.replace('{EXAMPLE}', EXAMPLE)
        prompt = prompt.replace('{EXAMPLE_OUTPUT}', EXAMPLE_OUTPUT)
        prompt = prompt.replace('{OUTPUT_START}', OUTPUT_START)
        prompt = prompt.replace('{text}', text)

        return prompt
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt from {filepath}: {e}")
        raise


def get_prompts(text: str, num_prompts: int = 5) -> List[str]:
    """Get prompts for paraphrasing by loading from files

    Loads syntactic rewrite prompts that preserve grammatical structure while replacing vocabulary.
    5 prompt files available: syntactic_rewrite_1.txt ~ syntactic_rewrite_5.txt

    Args:
        text: Text to paraphrase
        num_prompts: Number of prompts to generate (max 5)

    Returns:
        List of formatted syntactic rewrite prompts loaded from files
    """
    templates = []
    for i in range(1, min(num_prompts, 5) + 1):
        filename = f'syntactic_rewrite_{i}.txt'
        templates.append(load_prompt_from_file(filename, text))

    return templates


def validate_paraphrase_format(text: str) -> bool:
    """Validate that paraphrased text is not empty

    Args:
        text: Paraphrased text to validate

    Returns:
        True if format is valid (non-empty), False otherwise
    """
    return bool(text and text.strip())


def clean_paraphrase(text: str) -> str:
    """Clean up generated paraphrase text

    Handles cases where the model repeats the entire prompt template in its output.

    Args:
        text: Raw paraphrase text

    Returns:
        Cleaned paraphrase text
    """
    original_text = text

    # Check if the text contains the full prompt template
    # Pattern: "Task: Rewrite the following..." indicates full prompt repetition
    if "Task: Rewrite" in text or "You are a linguistic rewriter" in text:
        # The prompt template has been repeated. Extract the actual paraphrase.

        # Strategy 1: Look for "Rewritten:" marker after the last "Original:"
        if "Rewritten:" in text:
            # Split by "Rewritten:" and take the last occurrence
            parts = text.split("Rewritten:")
            if parts:
                text = parts[-1].strip()

        # Strategy 2: If no "Rewritten:" marker, split by "Original:"
        elif "Original:" in text:
            parts = text.split("Original:")
            # The structure is: ... Original: [example] ... Original: [input_text] [paraphrase]
            # We want the content after the LAST "Original:"
            if len(parts) >= 2:
                # Get text after last "Original:"
                after_last_original = parts[-1].strip()

                # The paraphrase might be on a new line or continue after the original text
                # Remove the "Rewritten:" prefix if it exists
                after_last_original = re.sub(r'^Rewritten:\s*', '', after_last_original, flags=re.IGNORECASE)

                # If text contains newlines, the paraphrase is likely after the first newline
                if '\n' in after_last_original:
                    lines = after_last_original.split('\n', 1)
                    if len(lines) > 1 and lines[1].strip():
                        text = lines[1].strip()
                    else:
                        text = after_last_original
                else:
                    text = after_last_original

        # Strategy 3: Extract everything after the example section
        else:
            # Look for "Example:" section end
            if "Example:" in text:
                # Find content after the example section
                match = re.search(r'Example:.*?(?:Rewritten:|Output:)\s*([^\n]+)', text, re.DOTALL)
                if match:
                    # Get everything after the example
                    remaining = text[match.end():]
                    if remaining.strip():
                        text = remaining.strip()

    # Remove common artifacts and prefixes
    artifacts = [
        "Rephrase:", "Rephrased:", "Rephrased text:",
        "Rewritten text:", "Rewritten:",
        "Paraphrase:", "Paraphrased:", "Paraphrased text:",
        "Different version:", "Output:", "Text:", "→",
        "Task:", "Original:"  # Remove these if they still exist
    ]

    for artifact in artifacts:
        if text.startswith(artifact):
            text = text[len(artifact):].strip()
        text = text.replace(artifact, " ").strip()

    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Final validation: if cleaned text is too short or looks wrong, return original
    if not text or len(text) < 3:
        return original_text.strip()

    return text


def generate_single_paraphrase(
    text: str,
    model,
    tokenizer,
    prompt: str,
    use_beam: bool,
    num_beams: int,
    temperature: float = 0.2,
    top_p: float = 0.9,
    use_sampling: bool = False
) -> Optional[str]:
    """Generate a single paraphrase with controlled sampling

    Decoding Strategy:
    - temperature: 0.0-0.3 (sharp distribution for structural consistency)
    - top_p: 0.9-0.95 (controlled diversity)
    - do_sample: True with low temperature (lexical diversity + syntax preservation)
    - max_new_tokens: ±10% of original length (prevent length bias)

    Args:
        text: Original text
        model: Language model
        tokenizer: Tokenizer
        prompt: Prompt to use
        use_beam: If True, use beam search; if False, use sampling
        num_beams: Number of beams for beam search (1-2 recommended)
        temperature: Temperature for sampling (default: 0.2)
        top_p: Top-p (nucleus) sampling parameter (default: 0.9)
        use_sampling: If True, use controlled sampling; if False, use beam

    Returns:
        Paraphrased text or None if generation failed
    """
    # Tokenize
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Calculate target length (±10% of original)
    original_token_count = len(tokenizer.tokenize(text))
    max_new_tokens = int(original_token_count * 1.1) + 15
    min_new_tokens = max(10, int(original_token_count * 0.9))

    with torch.no_grad():
        if use_beam:
            # Minimal beam search (1-2 beams for stability)
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=min(num_beams, 2),  # Cap at 2 beams
                num_return_sequences=1,
                do_sample=False,
                early_stopping=False,  # Prevent premature truncation
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05  # Minimal penalty to avoid lexical artifacts
            )
        elif use_sampling:
            # Controlled sampling: low temperature + nucleus sampling
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=temperature,  # 0.2 for sharp distribution
                top_p=top_p,  # 0.9 for controlled diversity
                top_k=50,  # Limit extreme choices
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05
            )
        else:
            # Deterministic sampling with low temperature
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=0.2,  # Very low for near-deterministic behavior
                top_p=0.9,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05
            )

    # Decode
    input_length = inputs.input_ids.shape[-1]
    generated_tokens = outputs[0][input_length:]
    paraphrase_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    paraphrase_text = clean_paraphrase(paraphrase_text)

    # Validate length and format
    if paraphrase_text and len(paraphrase_text.split()) >= 3:
        if validate_paraphrase_format(paraphrase_text):
            return paraphrase_text

    return None


def generate_with_beam_search(
    text: str,
    model,
    tokenizer,
    prompt: str,
    num_paraphrases: int,
    num_beams: int,
    return_multiple: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.9,
    use_soft_sampling: bool = False
) -> List[Dict[str, Any]]:
    """Generate paraphrases using controlled beam search

    Strategy:
    - temperature: 0.2 (sharp distribution, minimal noise)
    - top_p: 0.9 (controlled diversity)
    - num_beams: 1-2 (minimal beam search, avoid over-optimization)
    - do_sample: True with low temperature (deterministic + lexical diversity)
    - length control: ±10% of original (prevent simplification bias)

    Args:
        text: Original text
        model: Language model
        tokenizer: Tokenizer
        prompt: Prompt to use
        num_paraphrases: Number of paraphrases to generate
        num_beams: Number of beams (capped at 2 for stability)
        return_multiple: If True, return multiple beam results
        temperature: Temperature for controlled sampling (default: 0.2)
        top_p: Top-p for nucleus sampling (default: 0.9)
        use_soft_sampling: If True, use controlled sampling (recommended)

    Returns:
        List of paraphrases
    """
    try:
        # Tokenize
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Calculate target length (±10% of original)
        original_token_count = len(tokenizer.tokenize(text))
        max_new_tokens = int(original_token_count * 1.1) + 15
        min_new_tokens = max(10, int(original_token_count * 0.9))

        with torch.no_grad():
            # Beam search: return multiple sequences
            num_return = min(num_paraphrases, min(num_beams, 2)) if return_multiple else 1

            if use_soft_sampling:
                # Controlled sampling: low temperature + nucleus sampling
                logger.info(f"Using controlled sampling (temperature={temperature}, top_p={top_p})")
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    num_beams=min(num_beams, 2),  # Cap at 2 beams
                    num_return_sequences=num_return,
                    do_sample=True,  # Enable sampling
                    temperature=temperature,  # 0.2 for sharp distribution
                    top_p=top_p,  # 0.9 for controlled diversity
                    top_k=50,  # Limit extreme choices
                    early_stopping=False,  # Prevent premature truncation
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05  # Minimal penalty
                )
            else:
                # Minimal beam search (deterministic)
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    num_beams=min(num_beams, 2),  # Cap at 2 beams
                    num_return_sequences=num_return,
                    do_sample=False,
                    early_stopping=False,  # Prevent premature truncation
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05  # Minimal penalty
                )

        # Extract paraphrases from beam outputs
        input_length = inputs.input_ids.shape[-1]
        paraphrases = []

        for idx, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            paraphrase_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            paraphrase_text = clean_paraphrase(paraphrase_text)

            if paraphrase_text and len(paraphrase_text.split()) >= 3:
                if validate_paraphrase_format(paraphrase_text):
                    paraphrases.append({
                        'text': paraphrase_text,
                        'score': 1.0 / (idx + 1),
                        'method': 'controlled_beam',
                        'beam_idx': idx
                    })

        logger.debug(f"Generated {len(paraphrases)} valid paraphrases using controlled beam search")
        return paraphrases[:num_paraphrases]

    except Exception as e:
        logger.error(f"Beam search failed: {e}")
        return [{'text': text, 'score': 1.0, 'method': 'original'}]


def generate_with_multiple_prompts(
    text: str,
    model,
    tokenizer,
    prompts: List[str],
    use_beam: bool,
    num_paraphrases: int,
    num_beams: int,
    temperature: float = 0.2,
    top_p: float = 0.9,
    use_sampling: bool = False
) -> List[Dict[str, Any]]:
    """Generate paraphrases using multiple prompts with controlled sampling

    Strategy:
    - temperature: 0.2 (default, sharp distribution)
    - top_p: 0.9 (controlled diversity)
    - use_sampling: True recommended (deterministic + lexical diversity)

    Args:
        text: Original text
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to use
        use_beam: If True, use beam search; if False, use controlled sampling
        num_paraphrases: Number of paraphrases to generate
        num_beams: Number of beams for beam search (capped at 2)
        temperature: Temperature for sampling (default: 0.2)
        top_p: Top-p (nucleus) sampling parameter (default: 0.9)
        use_sampling: If True, use controlled sampling (recommended)

    Returns:
        List of paraphrases
    """
    paraphrases = []
    if use_sampling:
        method = 'controlled_sampling_Nprompts'
    elif use_beam:
        method = 'controlled_beam_Nprompts'
    else:
        method = 'deterministic_Nprompts'

    for prompt_idx, prompt in enumerate(prompts):
        try:
            paraphrase = generate_single_paraphrase(
                text, model, tokenizer, prompt, use_beam, num_beams,
                temperature=temperature, top_p=top_p, use_sampling=use_sampling
            )

            if paraphrase:
                paraphrases.append({
                    'text': paraphrase,
                    'score': 1.0,
                    'method': method,
                    'prompt_idx': prompt_idx
                })

        except Exception as e:
            logger.warning(f"Failed to generate paraphrase with prompt {prompt_idx}: {e}")
            continue

    logger.debug(f"Generated {len(paraphrases)} paraphrases using {method}")
    return paraphrases[:num_paraphrases]
