import subprocess

def ask_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3:8b"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

response = ask_llm("Say hello in one sentence")
print(response)