HomeMatch Project
=================

This consists of a self-contained `HomeMatch.ipynb` notebook that sets up the environment 
and generates 12 augmented listings as required.

Put your valid OpenAI Platform API key in a local `.env` file before running:
OPENAI_API_KEY = <your API key>

(NOTE: I don't think the vocarium api key is compatible with the latest openai libraries, so I used my own key for testing)

I chose to use GPT-4 and the latest versions of opaenai client, langchain, chromadb as of today (8/26/2025)
My local python 3.10 virtual environment contains:

langchain                                0.3.27
langchain-chroma                         0.2.5
langchain-core                           0.3.75
langchain-openai                         0.3.32
openai                                   1.102.0
pip                                      25.2
pydantic                                 2.11.7
pydantic_core                            2.33.2

Assuming this environment and a suitable OPENAI_API_KEY, just run the notebook ... it should output the listings.txt 
file and various messages to the console along the way.

You can edit the last cell to vary the user preferences (either hard coded or interactive inputs)

