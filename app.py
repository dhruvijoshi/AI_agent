# Bring in deps
import os 
from apikey import apikey 
os.environ['OPENAI_API_KEY'] = apikey
import streamlit as st 
# from langchain.llms import OpenAI
from langchain_openai import OpenAI
llm = OpenAI(model_name='text-davinci-003', temperature=0.7, max_tokens=512)
from langchain.llms import AzureOpenAI

from langchain_community.chat_models.huggingface import ChatHuggingFace

import langchain.llms 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

# error handling for open api key
# if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
#     st.error("Please set the OPENAI_API_KEY environment variable.")
#     st.stop()


# App framework
st.title('DJ GPT')
prompt = st.text_input('Write topic to generate an essay') 



# Prompt templates 
# Only need to provide topic to write essay
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me an essay about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    # input_variables = ['title'], 
    template='write me an essay on TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
    
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
# sequential_chain = SequentialChain(chains=[title_chain, script_chain], verbose=True)


wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    # response = sequential_chain.run(prompt)
    # response = llm(prompt)
    # st.write(response)
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)