from ctransformers import AutoModelForCausalLM
import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    cl.user_session.set("message_history", [])

    await cl.Message("Model initialized. How can I help you?").send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    message_history = cl.user_session.get("message_history")
    prompt = get_prompt(message.content, message_history)
    answer = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        answer += word
    message_history.append(answer)

    await msg.update()


def get_prompt(question: str, history: list[str] = []) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{question}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt
