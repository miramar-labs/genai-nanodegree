#!/usr/bin/env python
# coding: utf-8

# # Custom Chatbot Project

# TODO: In this cell, write an explanation of which dataset you have chosen and why it is appropriate for this task
# 
# For this project I will implement a custom chatbot with specialized knowledge of Worlds events for 2025. I will use the 2025 Wikipedia web page for this purpose.

# In[44]:


import openai
openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = "voc-2027486366126677406145368181200542799.11491375"


# ## Data Wrangling
# 
# TODO: In the cells below, load your chosen dataset into a `pandas` dataframe with a column named `"text"`. This column should contain all of your text data, separated into at least 20 rows.

# In[45]:


import requests

# Get the Wikipedia page for "2025" since OpenAI's models stop in 2021
params = {
    "action": "query",
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2025",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
response_dict["query"]["pages"][0]["extract"].split("\n")


# In[46]:


import pandas as pd

# Show more columns and set column width
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.width', None)              # Auto-detect best width
pd.set_option('display.max_colwidth', None)       # Show full content of each cell

# Load page text into a dataframe
df = pd.DataFrame()
df["text"] = response_dict["query"]["pages"][0]["extract"].split("\n")


# In[47]:


from dateutil.parser import parse

# Clean up text to remove empty lines and headings
df = df[(df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))]

# In some cases dates are used as headings instead of being part of the
# text sample; adjust so dated text samples start with dates
prefix = ""
for (i, row) in df.iterrows():
    # If the row already has " - ", it already has the needed date prefix
    if " – " not in row["text"]:
        try:
            # If the row's text is a date, set it as the new prefix
            parse(row["text"])
            prefix = row["text"]
        except:
            # If the row's text isn't a date, add the prefix
            row["text"] = prefix + " – " + row["text"]
df = df[df["text"].str.contains(" – ")].reset_index(drop=True)


# In[48]:


df


# ## Custom Query Completion
# 
# TODO: In the cells below, compose a custom query using your chosen dataset and retrieve results from an OpenAI `Completion` model. You may copy and paste any useful code from the course materials.

# In[49]:


import openai
import pandas as pd
import time

# Assume you already have sentences_df with a 'text' column
# Set your model
EMBEDDING_MODEL = "text-embedding-3-small"

# Batching parameters
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request for `text-embedding-3-*`

# Helper: Call OpenAI and handle retries
def get_embeddings_batch(text_list, model=EMBEDDING_MODEL):
    try:
        response = openai.Embedding.create(input=text_list, engine=model)
        return [e.embedding for e in response["data"]]
    except Exception as e:
        print("Retrying after error:", e)
        time.sleep(2)
        return get_embeddings_batch(text_list, model=model)

# Run batching
all_embeddings = []

for i in range(0, len(df), BATCH_SIZE):
    batch_texts = df["text"].iloc[i:i+BATCH_SIZE].tolist()
    print(f"Embedding batch {i//BATCH_SIZE + 1} of {len(df)//BATCH_SIZE + 1}...")
    batch_embeddings = get_embeddings_batch(batch_texts)
    all_embeddings.extend(batch_embeddings)

# Add to DataFrame
df["embeddings"] = all_embeddings


# In[50]:


def get_embedding_for_userQ(question):

  # Generate the embedding response
  response = openai.Embedding.create(input=question, engine=EMBEDDING_MODEL)

  # Extract the embeddings from the response
  return response.data[0].embedding


# In[51]:


def cosine_similarity(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# In[52]:


# calculate cosine distances between userQ and each embedding in df, then sort
def get_rows_sorted_by_relevance(q, df):
  Qe = get_embedding_for_userQ(q)
  distances = [1-cosine_similarity(e, Qe) for e in df["embeddings"]]
  df["distances"] = distances
  df.sort_values(by="distances", ascending=True, inplace=True)
  return df


# In[53]:


import tiktoken


# In[54]:


def create_prompt_basic(question, df, max_token_count):
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
Answer the question below, and if the question
can't be answered, say "I don't know"

---

Question: {}
Answer:"""

    current_token_count = len(tokenizer.encode(prompt_template)) + \
                            len(tokenizer.encode(question))
    
    if current_token_count > max_token_count:
       raise Exception("max_token_count exceeded... aborting...")

    return prompt_template.format(question)


# In[55]:


def create_prompt_enhanced(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context:

{}

---

Question: {}
Answer:"""

    current_token_count = len(tokenizer.encode(prompt_template)) + \
                            len(tokenizer.encode(question))

    if current_token_count > max_token_count:
       raise Exception("max_token_count exceeded... aborting...")
    
    context = []
    for text in get_rows_sorted_by_relevance(question, df)["text"].values:

        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)


# In[56]:


COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"

def answer_question_enhanced(question, df, max_prompt_tokens=1800, max_answer_tokens=150):
  try:
      prompt = create_prompt_enhanced(question, df, max_prompt_tokens)

      response = openai.Completion.create(
          model=COMPLETION_MODEL_NAME,
          prompt=prompt,
          max_tokens=max_answer_tokens
      )
      return response["choices"][0]["text"].strip()
  except Exception as e:
      print(e)
      return ""


# In[57]:


COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"

def answer_question_basic(question, df, max_prompt_tokens=1800, max_answer_tokens=150):
  try:
      prompt = create_prompt_basic(question, df, max_prompt_tokens)

      response = openai.Completion.create(
          model=COMPLETION_MODEL_NAME,
          prompt=prompt,
          max_tokens=max_answer_tokens
      )
      return response["choices"][0]["text"].strip()
  except Exception as e:
      print(e)
      return ""


# ## Custom Performance Demonstration
# 
# TODO: In the cells below, demonstrate the performance of your custom query using at least 2 questions. For each question, show the answer from a basic `Completion` model query as well as the answer from your custom query.

# ### Question 1

# In[58]:


print(answer_question_basic("What is Deepseek?", df));


# In[59]:


print(answer_question_enhanced("What is Deepseek?", df));


# ### Question 2

# In[60]:


print(answer_question_basic("Who is the current Prime Minister of Canada?", df));


# In[61]:


print(answer_question_enhanced("Who is the current Prime Minister of Canada?", df));


# In[ ]:





# In[ ]:




