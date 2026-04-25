from typing import Dict, Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from init_llm import ark_llm


@tool
def query_customer_data(customer_id: str) -> Dict[str, Any]:
    """
    查询客户基本信息
    Args:
        customer_id: 客户ID，用于唯一标识客户
    Returns:
        包含客户基本信息的字典，如姓名、等级、加入日期等
    """
    # 模拟数据库查询
    return {"customer_id": customer_id,"name": "张三","level": "VIP","join_date": "2023-01-15"}


@tool
def check_order_history(customer_id: str) -> Dict[str, Any]:
    """
    查询客户订单历史
    Args:
        customer_id: 客户ID，用于唯一标识客户
    Returns:
        包含客户订单历史的字典，如总订单数、总花费等
    """
    return {"customer_id": customer_id,"total_orders": 15,"total_spent": 25800.00}


@tool
def get_current_promotions() -> Dict[str, Any]:
    """
    获取当前可用促销活动
    Returns:
        包含当前可用促销活动的字典，如活动名称、有效日期等
    """
    return {
        "promotions": ["老用户优惠", "会员专属折扣"],
        "valid_until": "2027-01-31"
    }

agent = create_agent(
    model = ark_llm,
    system_prompt="你是一个客户服务助手，负责回答客户关于订单、促销活动等问题。",
    tools=[query_customer_data, check_order_history, get_current_promotions],
    checkpointer= InMemorySaver()# 可以将状态保存到内存、数据库等
)

config= {"configurable": {"thread_id": "xxx"}}

for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "查询客户ID为12345的完整信息和可用优惠活动"}]},
        config=config,
        # stream_mode="values" #每一步都输出完整状态
        # stream_mode="updates" #每一步只输出增量状态 默认模式
        # stream_mode="messages" #按token输出 （打字机效果）
        #stream_mode="tasks" #按任务形式输出，多了任务信息
        #="debug" #和task 类似，多了时间戳等信息
        #stream_mode="checkpoints" #当检查点被创建时会触发输出 输出包含检查点中的状态
        stream_mode="checkpoints" #当检查点被创建时会触发输出 输出包含检查点中的状态
):
    print(chunk)
    print("-"*50)

    # for tp, data in chunk.items():
    #     data["messages"][-1].pretty_print()

    # for tp, data in chunk.items():
    #     if tp =="model":
    #         print("大模型回复：")
    #         print(data["messages"][-1].content)
    #     if tp == "tools":
    #         print("工具回复：")
    #         print(data["messages"][-1].content)
    #     print("-" * 50)

