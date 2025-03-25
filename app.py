import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

## Set upi the Stramlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Gemma2-9b-It")

wikiwrapper = WikipediaAPIWrapper()
wikitool = Tool(
    name="Wikipedia",
    func=wikiwrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"
)

math_chain = LLMMathChain.from_llm(llm)
math_tool=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

system_prompt = """
You are an agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:
"""
prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{question}")
])
chain = LLMChain(llm=llm,prompt=prompt)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

assistant_agent = initialize_agent(
    tools=[math_tool,wikitool,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"ai","content":"Welcome to math solver"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

query = st.chat_input("Enter your message")

if query:    
    with st.spinner("Generating response..."):
        st.chat_message("human").write(query)
        st.session_state["messages"].append({"role":"human","content":query})

        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

        st.chat_message("ai").write(response)
        st.session_state["messages"].append({"role":"ai","content":response})


# st.write(math_chain.run("20+30"))
# st.write(math_chain.invoke({"question": "What is 25 squared?"}))
# st.write(math_chain.apply([{"question": "10 + 5"}, {"question": "8 * 7"}]))
