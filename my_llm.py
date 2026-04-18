"""
创建各类大模型
"""

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ARK_API_KEY, ARK_BASE_URL

#model class 方法初始化模型
deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL
)

#model class 方法初始化模型
ark_llm = ChatOpenAI(
    model="ark-code-latest",
    api_key=ARK_API_KEY,
    base_url=ARK_BASE_URL
)
