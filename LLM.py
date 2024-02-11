import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

df_full = pd.read_csv('tennis_data.csv')
"""
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df_full,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
"""

df_trimmed = df_full
df_trimmed['player_1_points'] = df_trimmed['player_1_points'].replace(['AD'], 60)
df_trimmed['player_2_points'] = df_trimmed['player_2_points'].replace(['AD'], 60)

df_trimmed["player_1_points"] = pd.to_numeric(df_trimmed["player_1_points"])
df_trimmed["player_2_points"] = pd.to_numeric(df_trimmed["player_2_points"])
df_trimmed.info()


df_trimmed = df_trimmed.groupby(['game_id'], as_index=False).agg({
    'player_1_points':'sum', 'player_2_points':'sum','player_1_games':'sum',
     'player_2_games':'sum', 'player_1_sets':'sum', 'player_2_sets':'sum',
       'match_winning_player':'mean'
})

# Try it out! You can use my API key lol. Ask it something like 'who won match 135' if using trimmed, or who scored the most points
# if using the full dataset
print('------------------------------------------------------------------------------------------------')
print("🎾Hi! I'm the tennis bot. Ask a question about the full set of matches or the trimmed set.🎾")
print('------------------------------------------------------------------------------------------------')
set = input('Please select a dataset: (f for full, t for trimmed): ')
query = input('Please ask a question about the dataset (type "quit" or "q" to exit): ')
while query != 'quit' and query != 'q':
    if set == 'f':
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        df_full,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    elif set == 't':
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        df_trimmed,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    print('🎾 is thinking ...')
    print(agent.invoke(query))
    set = input('Please select a dataset: (f for full, t for trimmed): ')
    query = input('Please ask a question about the dataset (type "quit" or "q" to exit): ')