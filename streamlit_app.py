from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

# ===
# Import packages: 
import numpy as np 
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import nltk
from os import listdir
from os.path import isfile, join
from wordcloud import WordCloud
from operator import itemgetter
import random
import community as com
import os
from fa2 import ForceAtlas2
from matplotlib.pyplot import figure
import powerlaw
import networkx.algorithms.community as nx_comm
import community as community_louvain
import json
import collections
from nltk.tokenize import sent_tokenize,word_tokenize, RegexpTokenizer 
from labMTsimple.storyLab import emotionFileReader, emotion, stopper, emotionV
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import itertools
import urllib
import imdb
# ===

"""
# Welcome to the project website for the course Social graphs and interactions (02805)
# Made by Mihaela-Elena Nistor, Viktor Anzhelev Tsanev and Jacob Kofod

"""

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Read characters from CSV
characters_path = "nodes.csv"
df_characters = pd.read_csv(characters_path)

print(df_characters)

data_load_state.text('Loading data...done!')

# ====
# Create undirected graph
G = nx.Graph()

# Read nodes from csv given as part of the exercises
path_nodes = "nodes.csv"
df_nodes = pd.read_csv(path_nodes)

#print(df_nodes)

# Add nodes to the graph
for index, character in df_nodes.iterrows():
    if (character["Name"]):
        G.add_node(character["Name"])

# Read edges from csv file. Edges where to and from referes to the same
# node was been removed.
path_edges = "edges.csv"
df_edges = pd.read_csv(path_edges)

# Load episodes
path_episodes = "episodes.csv"
df_episodes = pd.read_csv(path_episodes)

#H.add_nodes_from(list_of_characters)
friends_links = list(zip(df_edges.From, df_edges.To))
G.add_edges_from(friends_links)
# ====