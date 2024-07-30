from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm.send import Messages


class LcChatWrapper(Runnable):
    def __init__(self, send_fn) -> None:
        self.send_fn = send_fn

    def invoke(self, input, config):
        messages = self.to_messages(input)
        output = self.send_fn(messages, {}).output
        print(config)

        return HumanMessage(
            content=output,
        )

    def to_messages(
        self,
        invoke_input,
    ) -> Messages:
        messages = []
        for m in invoke_input.messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                raise NotImplementedError("")
            messages.append(
                {
                    "role": role,
                    "content": m.content,
                }
            )
        return messages


class LcTextEmbWrapper:
    def __init__(self, text_emb_send_fn) -> None:
        self.text_emb_send_fn = text_emb_send_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return self.text_emb_send_fn(texts)

    def embed_query(self, text: str) -> list[float]:

        return self.text_emb_send_fn([text])[0]
