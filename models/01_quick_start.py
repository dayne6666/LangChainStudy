"""
创建一个Agent 调用工具回答用户的问题
"""
from langchain.agents import create_agent
from langchain_core.tools import tool

from my_llm import deepseek_llm, ark_llm


@tool
def get_weather(location: str) -> str:
    """
    获取指定位置的天气信息
    """
    return f"天气信息：{location}的天气是晴朗"


agent = create_agent(
    model=ark_llm,
    tools=[get_weather],
    system_prompt="你是一个智能助手,尝试使用工具回答用户的问题"
)
# 调用agent
resp = agent.invoke({"messages": [{"role": "user", "content": "南京的天气怎么样"}]})
print(type(resp))
print(resp)