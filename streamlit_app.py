from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit
import streamlit as st
import logic as logic
# ===
# Import packages:
import numpy as np 
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from wordcloud import WordCloud
from operator import itemgetter
import random
import community as com
import os
from fa2 import ForceAtlas2
from matplotlib.pyplot import figure
st.set_page_config(page_title="My App",layout='wide')
# ===

"""
# Welcome to the project website for the course Social graphs and interactions (02805)
# Made by Mihaela-Elena Nistor, Viktor Anzhelev Tsanev and Jacob Kofod

"""

################################################################## Read characters from CSV###################################################################################################################
characters_path = "nodes.csv"
df_characters = pd.read_csv(characters_path)

path_nodes = "nodes.csv"
df_nodes = pd.read_csv(path_nodes)
path_edges = "edges.csv"
df_edges = pd.read_csv(path_edges)
##################################################################################CREATE NETWORK#################################################################################################################
G = logic.generate_network(df_nodes, df_edges)
with st.container():

    # === Video
    st.header('Introduction video')
    st.video('https://youtu.be/_qKCQAbOt_8')

###################################################################################### Basic stats###############################################################################################################
with st.container():


    st.header('Basic stats')

    numberOfNodes = len(G.nodes())
    numberOfEdges = len(G.edges())

    t = "The network has " + str(numberOfNodes) + " nodes and " + str(numberOfEdges) + " edges."
    st.text(t)


##################################################################### Degree distribution############################################################################################################################
with st.container():

    st.header('Degree distributions')
    st.pyplot(logic.generate_degree_distribution_plot(G))


##############################################################################WORDCLOUDS ################################################################################################################################


# Read words for characters from CSV
wordlistPath = "characterwords.csv"
df_wordlist = pd.read_csv(wordlistPath)

with st.container():

    st.header('Word cloud drawings')

    ch_selected = st.selectbox('I want to see the wordcloud for:', list(df_wordlist.Name.unique()))

    st.pyplot(logic.generate_wordcloud(ch_selected,df_wordlist))

# Gigant connected
largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc)

###################################################################################################### PLOT NETWORK #########################################################################################################
with st.container():

    st.header('Graph drawing')
    st.pyplot(logic.generate_graph(GCC))

###################################################################################################### PLOT NETWORK #########################################################################################################
episodes_sentiment = pd.read_csv('df_episode_sentiment.csv')
director_sentiment = pd.read_csv('director_sent_score.csv')
sentiment_of_pairs = pd.read_csv('sentiment_of_pairs.csv')
with st.container():

    col1, col2 = st.columns(2)

    with col1:
        col1.header('Sentiment for pairs of characters')
        logic.create_sentiment_graph(sentiment_of_pairs)

    with col2:
        col2.header('Sentiment for each director')
        logic.create_sentiment_graph(director_sentiment)

    st.header('Evolution of sentiment in time')
    logic.create_sentiment_graph(episodes_sentiment)