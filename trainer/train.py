import argparse
import logging
import warnings

import yaml
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

from trainer.model import GPT2FromScratch
from trainer.load_data import load_data


def compute_metrics(eval_pred) -> dict:
    """Function to calculate perplexity for current training

    Args:
        eval_pred: data from compute_metrics Trainer API

    Returns:
        dict: returns loss and perplexity
    """
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    try:
        perplexity = torch.exp(torch.tensor(loss)).item()
    except OverflowError:
        perplexity = float("inf")
    return {"loss": loss, "perplexity": perplexity}


# Set logging level
parser = argparse.ArgumentParser()
parser.add_argument(
    "-log",
    "--loglevel",
    default="warning",
    help="Provide logging level. Example --loglevel debug, default=warning",
)

args = parser.parse_args()
logging.basicConfig(level=args.loglevel.upper())
logging.info(f"Logging level {args.loglevel.upper()}.")
warnings.filterwarnings("ignore")

# reading config
with open("trainer/config.yaml", "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

# loading model
logging.info("Started to loading model")
model = GPT2FromScratch(
    model_name=cfg["model"],
    min_context_len=cfg["dataconfig"]["min_context_len"],
    max_context_len=cfg["dataconfig"]["max_context_len"],
)
model.tokenizer.pad_token = model.tokenizer.eos_token
logging.info(f'{cfg["model"]} size: {model.model_size:.1f}M parameters')

# loading data
tokenized_datasets = model.prepare_data(*load_data())
data_collator = DataCollatorForLanguageModeling(model.tokenizer, mlm=False)
logging.info("Dataset prepared.")

# training
logging.info("Started to training.")

MODEL_NAME = cfg["model_name"]
args = TrainingArguments(hub_model_id=MODEL_NAME, **cfg["trainer_args"])
trainer = Trainer(
    model=model.model,
    tokenizer=model.tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics,
)
