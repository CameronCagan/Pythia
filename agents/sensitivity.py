import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def sensitivity_agent(backend, information, SOP):
    system= SystemMessage(content="You have expertise in both clinical knowledge and advanced prompt engineering technologies following this SOP: \n " + SOP)
    human= HumanMessage(content=f"""
      Read the provided example, find and output the evidence of why the note is a positive. \n{information}\n
    """)
    messages = [system, human]
    llm = backend
    response = llm.invoke(messages)
    answer = response.content
    return answer

def summarizer_sensitivity(backend, evidence, prompt, sop):
    system= SystemMessage(content="You are a helpful assistant.")
    human= HumanMessage(content=f"""
    1. Summarize the evidence. \n
    2. Add the summarized evidence to the guidelines.\n
    3. Improve the given prompt based on the expanded guidelines.\n
    Only output the improved prompt.\n
    evidence: {evidence} \n
    guidelines:{sop}\n
    prompt:{prompt}
    """)
    messages = [system, human]
    llm = backend 
    response = llm.invoke(messages)
    answer = response.content
    return answer

