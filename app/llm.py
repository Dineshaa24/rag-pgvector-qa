"""Answer generation from retrieved chunks, using any OpenAI-compatible chat model through LangChain."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.store import Hit

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite the passages you used like [1] or [2]. "
    "If the answer is not in the context, say you don't know. Keep answers short."
)


def build_messages(question: str, hits: list[Hit]) -> list:
    context = "\n\n".join(f"[{i}] ({h.source})\n{h.content}" for i, h in enumerate(hits, start=1))
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Context:\n{context}\n\nQuestion: {question}")]


class Answerer:
    def __init__(self, model: str, base_url: str | None, api_key: str | None):
        from langchain_openai import ChatOpenAI
        self.chat = ChatOpenAI(model=model, base_url=base_url, api_key=api_key or "not-needed", temperature=0)

    def answer(self, question: str, hits: list[Hit]) -> str:
        return self.chat.invoke(build_messages(question, hits)).content
