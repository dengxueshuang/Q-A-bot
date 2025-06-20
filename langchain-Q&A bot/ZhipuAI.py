from langchain.llms.base import LLM
from zhipuai import ZhipuAI
from langchain_core.messages.ai import AIMessage
from pydantic import PrivateAttr

class ChatGLM4(LLM):
    _client: ZhipuAI = PrivateAttr()
    history: list = []

    def __init__(self, zhipuai_api_key=None):
        super().__init__()
        self._client = ZhipuAI(api_key=zhipuai_api_key)
        self.history = []

    @property
    def _llm_type(self) -> str:
        return "ChatGLM4"

    def invoke(self, prompt, history=[]):
        if history is None:
            history = []
        history.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model="glm-4",
            messages=history
        )
        result = response.choices[0].message.content
        return AIMessage(content=result)

    def _call(self, prompt: str, stop: list = None) -> str:
        return self.invoke(prompt).content


    def stream(self, prompt, history=None):
        if history is None:
            history = []

        history.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="glm-4",
            messages=history,
            stream=True
        )

        for chunk in response:
            yield chunk.choices[0].delta.content

# ✅ 初始化
llm = ChatGLM4(zhipuai_api_key="af34b738e09d45d195b44b1a137e29e4.RMw58YDOkWTy3qFv")

# ✅ 普通完整输出
response = llm.invoke("请讲一个减肥的笑话")
print(response.content)

# ✅ 流式输出（可选）
# for chunk in llm.stream("请讲一个减肥的笑话"):
#     print(chunk, end="", flush=True)
