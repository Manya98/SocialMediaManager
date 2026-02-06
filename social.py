# pip install langchain langchain-openai langchain-community firecrawl-py python-dotenv

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent, Tool
from langchain_community.document_loaders import FireCrawlLoader

load_dotenv()

# ---- Tool ----
def research_topic(query: str) -> str:
    """
    Searches and scrapes web content about a topic using Firecrawl.
    """
    loader = FireCrawlLoader(
        api_key=os.environ["FIRECRAWL_API_KEY"],
        mode="search",   # important
        params={"query": query, "limit": 5}
    )
    docs = loader.load()

    # Flatten content for the LLM
    return "\n\n".join(doc.page_content for doc in docs)

research_tool = Tool(
    name="WebResearcher",
    func=research_topic,
    description="Searches the web and summarizes up-to-date information on a topic."
)

# ---- LLM ----
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

# ---- Agent ----
agent = create_structured_chat_agent(
    llm=llm,
    tools=[research_tool],
    system_prompt=(
        "You are a tech analyst and content strategist. "
        "Research topics carefully and write concise, engaging LinkedIn posts."
    ),
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[research_tool],
    verbose=True
)

# ---- Run ----
task = (
    "Research the latest AI agent trends in 2025 "
    "and create a professional but engaging LinkedIn post."
)

result = agent_executor.invoke({"input": task})
print(result["output"])
