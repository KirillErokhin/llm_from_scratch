# LLM from scratch

This repo contains my personal research project for training Large language model (LLM) from scratch on small amount of data.

HuggingFace model repositry, where you can try: [GPT2WAP](https://huggingface.co/Kasdeja23/GPT2WaP)

## Code

The training code based on [HuggingFace tutorial](https://huggingface.co/learn/nlp-course/en/chapter7/6) and using Trainer for train the model.

## Data preparation

In this approarch for data used [War and peace](https://www.gutenberg.org/cache/epub/2600/pg2600.txt) book. 

Preparation:
- Deleted information about a book from site.
- Splitted with `python` with `"\r\n\r\n"` pattern.
- No clearing punctuation marks and commas for dialogues.
- No augmentation.
- Context length is set too small, so model cant's see the full context of text.

## Model 

For the first attempt used model config based on GPT2 with 124.4M parameters (GPT2LMHeadModel). 

The full model you can check on [HuggingFace](https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel) and also model [paramerets](https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/gpt2#transformers.GPT2Config).

## Evaluation

For evaluation used [perplexity](https://huggingface.co/docs/transformers/perplexity) with cross-entropy loss. 

In this attempt perplexity is too big, so model just generating incoherence text but with book style, model is overfitted on data. How to solve this problem is in research, maybe it's better to reduce size of a model or choose another architecture. 

## Tarining, deploy and inference

For inference and training used a docker container:

- For training used RTX 2x4090 and accelerate.
- For inference used simple gradio script and loading model from HuggingFace repository, device set to CPU .

## How to use

For training and inference using Docker container. First you need to build docker image:

```bash
docker compose build
```

Then choose specific config for Train or Inference: 

### Training

For training model you need to create `.env` file with HuggingFace token, change `config.yaml` with user_name and model_name or what you want to change.

Then in CLI:

```bash
MODE=training LOGGING=<python-log-level> GPU=<GPU-devices> docker compose up
```

And wait for model training.

### Inference

You can change `config.yaml` with user_name and model_name to load and inference model. 

Then in CLI:

```bash
MODE=inference LOGGING=<python-log-level> docker compose up
```

Then open http://localhost:1235/ and try our model.

## TODO

- [ ] Another split rull for data.
- [ ] Data augmentation: translate, paraphrase, generate with another model
- [ ] Data augmentation: give more information about the world of book
- [ ] Try another context length
- [ ] Reduce size model GPT2
- [ ] Full lock seed
- [ ] Automate changing model hyperparameters
- [ ] Try another model 
- [ ] Try another tokenizer 
- [ ] Gradio with different generate parameters
- [ ] Refact code with pytorch implementation