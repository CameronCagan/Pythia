import logging
#import torch
#from client.localLLM import LocalLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


sop = """ Given the information that cognitive concern could be relayed from patients 
(e.g., "patient complains of memory problems"), from family members or friends 
(e.g., "Sister says she is worried about the patient's memory", or providers in the notes, phone logs, 
indications for imaging procedures or other tests, etc.)
Bear in mind that an episode of delirium alone does not qualify a patient as having cognitive concerns present. 
Additionally, consider the following elements: Medications: Search for mentions of:
Donepezil, Rivastigmine, Galantamine, Memantine
Diagnoses and Symptoms: Look for terms related to:
Dementia, Alzheimer's disease
Memory issues: Recall problems, forgetfulness, lost items
Cognitive assessments: MOCA, MMSE
Mental status changes
Behavioral Symptoms (if cognitive concerns are present): a) Antipsychotic medications:
Aripiprazole, Haloperidol, Clozapine, Olanzapine, Quetiapine, Risperidone, Ziprasidone b) Behavioral symptoms:
Agitation, Aggression, Delirium, Hallucinations, Wandering
Psychosis, Paranoia, Combativeness, Delusions, Hostility, Outbursts
Criteria for Mild Cognitive Impairment (MCI): Check if all of the following are met: A. Reported cognitive changes (by patient, informant, or clinician) B. Objective evidence of cognitive impairment (typically including memory) C. Preserved independence in functional abilities D. Not meeting criteria for dementia
Criteria for Dementia: Check if all of the following are met: A. Cognitive or behavioral symptoms interfering with daily functioning B. Decline from previous levels of functioning C. Symptoms not explained by delirium or major psychiatric disorders D. Cognitive impairment detected through history-taking and objective assessment E. Impairment in at least 2 of the following: i. Ability to acquire and remember new information ii. Reasoning and handling complex tasks iii. Visuospatial abilities iv. Language functions v. Changes in personality, behavior, or comportment
"""

def specificity_agent(backend, information, SOP):
    system= SystemMessage(content="You have expertise in both clinical knowledge and advanced prompt engineering technologies following this SOP: \n " + SOP)
    human= HumanMessage(content=f"""
      Read the provided example, find and output the evidence of why the note is a negative for this prompt. \n{information}\n
    """)
    messages = [system, human]
    llm = backend #LocalLLM(base_url="http://localhost:11484/v1", temperature=0.1, max_tokens = 2048)

        
    #if torch.cuda.is_available():
        #logging.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    #else:
        #logging.info(f"CPU Memory usage: {torch.cuda.memory_reserved() / 1e6} MB")
    
    # terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # print('generate outputs')
    response = llm.invoke(messages)
    answer = response.content
    return answer

# def extract_sentence(text):
#     parts = text.split('"')
#     if len(parts) > 2:
#         return parts[1]
#     return None

def summarizer_specificity(backend, evidence, prompt, sop):
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
    llm = backend #LocalLLM(base_url="http://localhost:11484/v1", temperature = 0.1, max_tokens = 2048)

    #if torch.cuda.is_available():
        #logging.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    #else:
        #logging.info(f"CPU Memory usage: {torch.cuda.memory_reserved() / 1e6} MB")
    
    # print('generate outputs')
    response = llm.invoke(messages)
    answer = response.content
    return answer