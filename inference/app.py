from inference.load_model import pipe
import gradio as gr


def greet(text, intensity):
    return pipe(text, num_return_sequences=1)[0]["generated_text"]


demo = gr.Interface(
    fn=greet,
    inputs=["text"],
    outputs=["text"],
)
