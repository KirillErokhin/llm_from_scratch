from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from datasets import Dataset, DatasetDict


class GPT2FromScratch:
    """A class for initializing and preparing a GPT-2 model for training from scratch.

    This class provides methods for initializing a GPT-2 language model with custom configurations,
    tokenizing datasets, and preparing the data for training and validation.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer to process input text into tokens.
        min_context_len (int): Minimum length of text segments for training.
        max_context_len (int): Maximum length of text segments for training.
        config (AutoConfig): Configuration object for the GPT-2 model.
        model (GPT2LMHeadModel): The GPT-2 model instance with a language modeling head.
        model_size (float): The size of the model in millions of parameters.

    Methods:
        tokenize(element: Dataset) -> dict:
            Processes the input dataset and returns tokenized output within specified context lengths.

        prepare_data(train: list, test: list) -> DatasetDict:
            Tokenizes and organizes training and validation data into a DatasetDict ready for training.
    """

    def __init__(self, model_name: str, min_context_len: int, max_context_len: int):
        """Initialize model

        Args:
            model_name (str): CLM base model untrained
            min_context_len (int): Minimal length of text in data
            max_context_len (int): Maximum length of text in data
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.min_context_len = min_context_len
        self.max_context_len = max_context_len

        self.config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=len(self.tokenizer),
            n_ctx=self.max_context_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.model = GPT2LMHeadModel(self.config)

        model_size = sum(t.numel() for t in self.model.parameters())
        model_size /= 1000**2
        self.model_size = model_size

    def tokenize(self, element: Dataset) -> dict:
        """Tokenizer for transformers DataCollator

        Args:
            element (Dataset): Prepared dataset from Datasets

        Returns:
            dict: input_ids with batch
        """
        outputs = self.tokenizer(
            element["content"],
            truncation=True,
            max_length=self.max_context_len,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length >= self.min_context_len and length <= self.max_context_len:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    def prepare_data(self, train: list, test: list) -> DatasetDict:
        """Preparing data (tokenizing) for training

        Args:
            train (list): list of strings data for train
            test (list): list of strings data for test

        Returns:
            DatasetDict: returning DatasetDict with train and test tokenized data
        """

        raw_datasets = DatasetDict(
            {
                "train": Dataset.from_dict({"content": train}).shuffle(),
                "valid": Dataset.from_dict({"content": test}).shuffle(),
            }
        )
        tokenized_datasets = raw_datasets.map(
            self.tokenize,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        return tokenized_datasets
