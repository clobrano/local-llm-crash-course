from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(question: str) -> str:
    system = "You are an AI assistant that follows instruction extremely well. Give short answers."
    prompt = f"### System:\n{system}\n\n### User:\n{question}\n\n### Response:\n"
    print(prompt, end="", flush=True)
    return prompt


question = "Which is the capital of India?"

for i in llm(get_prompt(question), stream=True):
    print(i, end="", flush=True)
print()
