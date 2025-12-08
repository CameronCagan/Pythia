import logging
#import torch
#from client.localLLM import LocalLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def evaluate_cognitive_concerns (backend, prompt, information):
    
    system= SystemMessage(content="You are a specialized medical agent.")
    human= HumanMessage(content=f"{prompt}\n\n{information}")
    messages = [system, human]
    # print(messages)
    
    llm = backend #LocalLLM(
      #base_url="http://localhost:11484/v1",
      #temperature=0.1,
      #max_tokens=131072)
    # print('input_ids done')
    #logging.info(f"Input token length: {input_ids.shape[-1]}, Input character length: {len(messages[1]['content'])}")
        
    #if torch.cuda.is_available():
        #logging.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    #else:
        #logging.info(f"CPU Memory usage: {torch.cuda.memory_reserved() / 1e6} MB")
    
    # terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # print('generate outputs')
    response = llm.invoke(messages)
    answer = response.content
    
    return answer


# from all_prompts import prompt0, expertp1, expertp2, expertp3, expertp4, agentp1, agentp2, agentp3

# # List of all prompt functions
# prompt_functions = [prompt0, expertp1, expertp2, expertp3, expertp4, agentp1, agentp2, agentp3]

# def evaluate_cognitive_concerns(information, tokenizer, model):
#     responses = {}  # Dictionary to store responses for each prompt

#     for i, prompt_func in enumerate(prompt_functions):
#         # Generate the messages for the current prompt
#         messages = prompt_func(information)

#         # Tokenize the input
#         input_ids = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(model.device)

#         # Define terminators
#         terminators = [
#             tokenizer.eos_token_id,
#             tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]

#         # Generate response
#         outputs = model.generate(
#             input_ids,
#             max_new_tokens=256,
#             eos_token_id=terminators,
#             do_sample=True,
#             temperature=0.1,
#             top_p=0.9,
#         )
#         response = outputs[0][input_ids.shape[-1]:]
#         answer = tokenizer.decode(response, skip_special_tokens=True)

#         # Store response in dictionary
#         responses[f"prompt{i}"] = answer
#         print(f"#{i} prompt done")

#     return responses  # Return responses for all prompts


