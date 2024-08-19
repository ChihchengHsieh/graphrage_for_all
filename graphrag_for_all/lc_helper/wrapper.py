from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from graphrag_for_all.llm.send import Messages, ModelArgs


class LcChatWrapper(Runnable):
    def __init__(self, send_fn) -> None:
        self.send_fn = send_fn
        self.last_messages = None

    def invoke(
        self,
        input,
        config,
    ):
        messages = self.to_messages(input)
        self.last_messages = messages
        output = self.send_fn(messages, {}).output

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


from graphrag_for_all.llm.create import get_send_fn, get_text_emb_send_fn


class LcTextEmbWrapper:
    def __init__(self, text_emb_send_fn) -> None:
        self.text_emb_send_fn = text_emb_send_fn

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

        emb_out = self.text_emb_send_fn(texts, {})
        return emb_out

    def embed_query(self, text: str) -> list[float]:

        return self.text_emb_send_fn([text], {})[0]
