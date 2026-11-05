import os
from dotenv import load_dotenv
from ollama import Client
from typing import Optional


load_dotenv()


class OllamaClient:
    def __init__(self, model: Optional[str] = None):
        self.client = Client(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")

    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=(
                "You answer questions using the provided document excerpts. "
                "Always produce a useful answer from the excerpts when possible. "
                "Never reply with a refusal if the excerpts contain relevant content."
            ),
            options={
                "temperature": 0.2,
            },
        )
        return response["response"]
