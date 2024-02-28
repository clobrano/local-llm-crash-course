from ctransformers import AutoModelForCausalLM
import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )

    await cl.Message("Model initialized. How can I help you?").send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content)
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)

    await msg.update()


def get_prompt(question: str) -> str:
    system = "You are an AI assistant that follows instruction extremely well. Give short answers."
    prompt = f"### System:\n{system}\n\n### User:\n{question}\n\n### Response:\n"
    return prompt
