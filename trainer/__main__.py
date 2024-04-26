from trainer.train import trainer
import os


HF_TOKEN = os.getenv("HF_TOKEN")

if __name__ == "__main__":
    trainer.train()
    trainer.push_to_hub(token=HF_TOKEN)
