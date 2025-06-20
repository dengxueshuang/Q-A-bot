from langchain_community.chat_models import ChatZhipuAI
import os

# 设置 ZhipuAI 的 API Key
os.environ["ZHIPUAI_API_KEY"] = "af34b738e09d45d195b44b1a137e29e4.RMw58YDOkWTy3qFv"

# 初始化模型
llm = ChatZhipuAI()

# 调用模型
response = llm.invoke("美国的首都是哪里？不需要介绍")
print(response.content)
