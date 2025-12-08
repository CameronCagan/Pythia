import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def evaluate_cognitive_concerns (backend, prompt, information):
    
    system= SystemMessage(content="You are a specialized medical agent.")
    human= HumanMessage(content=f"{prompt}\n\n{information}")
    messages = [system, human]
    
    llm = backend 
    response = llm.invoke(messages)
    answer = response.content
    
    return answer


