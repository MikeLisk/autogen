import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from openai import OpenAI
async def main():

    # model_client = OpenAIChatCompletionClient(
    #     model="gpt-3.5-turbo",
    #     api_key="sk-h"# Optional if you have an OPENAI_API_KEY env variable set.
    #     )
    model_client = OpenAIChatCompletionClient(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="sk-",
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        )
    # Create the primary agent.
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Please provide constructive feedback. You can only reply 'Approved' after the work has undergone at least 3 rounds of revisions. Before that, please provide specific revision suggestions and do not reply 'APPROVE'.  Please reply 'APPROVE' once your feedback has been addressed.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")


    team = RoundRobinGroupChat(
        [primary_agent, critic_agent],
        termination_condition=text_termination,  # Use the bitwise OR operator to combine conditions.
    )

    await Console(team.run_stream(task="写一首关于秋天的短诗"))

    
if __name__ == "__main__":
    asyncio.run(main())

