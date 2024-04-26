from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
import yaml
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

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


with open("trainer/config.yaml", "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

logging.info("Loading model")
model_path = cfg["trainer_args"]["output_dir"]
tokenizer = AutoTokenizer.from_pretrained(f"./{model_path}/")
model = GPT2LMHeadModel.from_pretrained(f"./{model_path}/")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    device="cpu",
)
