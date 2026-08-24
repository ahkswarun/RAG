from src.ollama_client import OllamaClient

llm = OllamaClient()
response = llm.generate("Who wrote Harry Potter?")
print(response)