from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
import os

# 设置 API Key（建议使用环境变量）
os.environ["ZHIPUAI_API_KEY"] = "af34b738e09d45d195b44b1a137e29e4.RMw58YDOkWTy3qFv"

# 构建 Prompt 模板
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文：（{topic}）")

# 初始化模型（使用官方推荐的 glm-3-turbo）
model = ChatZhipuAI(model="glm-3-turbo")

# 输出解析器
output_parser = StrOutputParser()

# 构建 LangChain Chain
chain = prompt | model | output_parser

# 执行调用
response = chain.invoke({"topic": "能量饮料推荐"})
print(response)
