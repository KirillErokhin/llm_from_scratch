# LLM from scratch

This repo contains my personal research project for training Large language model (LLM) from scratch on small amount of data.

## Code

The training code based on [HuggingFace tutorial](https://huggingface.co/learn/nlp-course/en/chapter7/6) and using Trainer for train the model.

## Data preparation

In this approarch for data used [War and peace](https://www.gutenberg.org/cache/epub/2600/pg2600.txt) book. 

Preparation:
- Deleted information about a book from site.
- Splitted with `python` with `"\r\n\r\n"` pattern.
- No clearing punctuation marks and commas for dialogues.
- No augmentation.

## Model 

For the first attempt used model config based on GPT2 with 124.4M parameters (GPT2LMHeadModel). 

The full model you can check on [HuggingFace](https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel) and also model [paramerets](https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/gpt2#transformers.GPT2Config).

## Evaluation

For evaluation used [perplexity](https://huggingface.co/docs/transformers/perplexity) with cross-entropy loss. 

In this attempt perplexity is too big, so model just generating incoherence text but with book style, model is overfitted on data. How to solve this problem is in research, maybe it's better to reduce size of a model or choose another architecture.

## Tarining, deploy and inference

For inference and training used a docker container:

- For training used RTX 2x4090.
- For inference used simple gradio script and loading last saved model in checkpoints directory.

Objective:
Your task is to train a Language Model (LLM) from scratch using the text of a provided book.  Your model should generate text that captures the style and thematic elements of the source material.

Requirements:
Data Preprocessing:
Download the text of the book provided in the link https://www.gutenberg.org/cache/epub/2600/pg2600.txt.
Prepare data
Choose an appropriate model architecture for the LLM. 
Implement the model using a deep learning framework of your choice (e.g., TensorFlow, PyTorch).
Train the model.
Evaluate the model’s performance using suitable metrics.
Implement a function to generate text from a prompt that demonstrates the model's ability to mimic the style and content of the book.
Upload your project to google colab and share a link