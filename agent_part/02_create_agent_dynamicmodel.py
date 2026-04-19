from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from langchain.chat_models import init_chat_model

from langchain_core.tools import tool

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ARK_API_KEY, ARK_BASE_URL


@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气为晴朗，25°C。"


basic_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

advanced_model = init_chat_model(
    model="ark-code-latest",
    model_provider="openai",
    api_key=ARK_API_KEY,
    base_url=ARK_BASE_URL,

)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    print("request : ", request)
    # 判断消息条数，如果小于3条使用 basic_model，否则使用advanced_model
    message_count = len(request.state['messages'])
    if message_count < 3:
        model = basic_model
    else:
        model = advanced_model

    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    tools=[get_current_location, get_weather],
    # 中间件等于java里的拦截器
    middleware=[dynamic_model_selection],
)

response = agent.invoke({"messages": [
    {"role": "system", "content": "你是一个天气助手"},
    {"role": "user", "content": "我现在在的位置天气如何？"}
]})
print(response)
