"""
Pile Dataset for Paraphrase Generation

This module provides a PyTorch Dataset class for loading and processing
text data from The Pile dataset.
"""

import torch
from datasets import load_dataset


class PileTextDataset(torch.utils.data.Dataset):
    """Dataset for Pile text data (not QA format, just raw text)

    This dataset loads text from The Pile dataset and prepares it for
    language model processing with proper tokenization and padding.

    Args:
        data_path (str): Path to the dataset (e.g., 'EleutherAI/pile')
        tokenizer: HuggingFace tokenizer
        model_family (str): Model family name (e.g., 'llama2-7b')
        max_length (int): Maximum sequence length for tokenization
        split (str): Dataset split (e.g., 'train', 'test')
        num_samples (int, optional): Number of samples to load (None for all)
        subset_name (str): Pile subset name (e.g., 'all', 'PubMed Abstracts')
    """

    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None,
                 num_samples=None, subset_name='all'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family

        print(f"Loading Pile dataset: {data_path}, split: {split}, subset: {subset_name}")

        if data_path.startswith('EleutherAI/pile'):
            self.data = load_dataset(data_path, subset_name, split=split, streaming=False)
        else:
            self.data = load_dataset('json', data_files=data_path, split=split)

        if num_samples and num_samples < len(self.data):
            self.data = self.data.select(range(num_samples))

        print(f"Loaded {len(self.data)} samples from Pile")

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset

        Returns:
            tuple: (input_ids, labels, attention_mask) as torch tensors
        """
        text = self.data[idx]['text']

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        # Pad to max_length
        pad_length = self.max_length - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [self.tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

        # Create labels (for language modeling)
        if len(encoded.input_ids) == self.max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [self.tokenizer.eos_token_id] + [-100] * (pad_length - 1)

        return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)
