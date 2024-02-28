from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

system = "You are an AI assistant that follows instruction extremely well. Give short answers."
instruction = "Which is the capital of India?"

prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
for i in llm(prompt, stream=True):
    print(i, end="", flush=True)
print()
