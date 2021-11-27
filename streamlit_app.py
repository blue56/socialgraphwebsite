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
from os import listdir
from os.path import isfile, join
from wordcloud import WordCloud
from operator import itemgetter
import random
import community as com
import os
from fa2 import ForceAtlas2
from matplotlib.pyplot import figure

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

friends_links = list(zip(df_edges.From, df_edges.To))
G.add_edges_from(friends_links)

# === Basic stats
st.header('Basic stats')

numberOfNodes = len(G.nodes())
numberOfEdges = len(G.edges())

t = "The network has " + str(numberOfNodes) + " nodes and " + str(numberOfEdges) + " edges."
st.text(t) 

# ==== Word clouds
st.header('Word cloud drawings')

# Read words for characters from CSV
wordlistPath = "characterwords.csv"
df_wordlist = pd.read_csv(wordlistPath)

colormapList = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

for champion_row in df_wordlist.iterrows():
    
    champion = champion_row[1][0]
    wordcloud_string = champion_row[1][1]

    # Create and generate a word cloud image:
    colormap = random.choice(colormapList)
    wordcloud = WordCloud(width=1000, height=1000, background_color='white', colormap=colormap).generate(wordcloud_string)
    wordcloud.collocations = False

    # Display the generated image:
    plt.figure( figsize=(15,7))
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(champion,fontsize=15)
    plt.show()
    st.pyplot(plt)


#### Degree distribution

st.header('Degree distributions')

#get all sentiment values 
degrees = [i[1] for i in list(G.degree)]
# get hist values and edges 
hist, bin_edges = np.histogram(degrees)

figure(figsize=(12, 10), dpi=80)
n, bins, patches = plt.hist(x=degrees, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Degree',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('$\it{F}$•$\it{R}$•$\it{I}$•$\it{E}$•$\it{N}$•$\it{D}$•$\it{S}$ - Degree Distribution',fontsize=18)
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.xticks(np.arange(min(degrees), max(degrees)+1, 10))
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

st.pyplot(plt)

######
# Plot network

st.header('Graph drawing')

largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc)

colors = []
for node in list(GCC.nodes()):
    if node == 'Ross Geller':
        colors.append("#00009E")
    elif node == 'Chandler Bing':
        colors.append("#FFF580")
    elif node == 'Joey Tribbiani':
        colors.append("#9A0006")
    elif node == 'Phoebe Buffay':
        colors.append("#42A2D6")
    elif node == 'Rachel Greene':
        colors.append("#FFDC00")
    elif node =='Monica Geller':
        colors.append("#FF4238")
    else:
        colors.append("#196F3D")


edge_color = []
for edge in list(GCC.edges()):
    
    a = edge[0]
    b = edge[1]
    if ('Chandler' in a) or ('Chandler' in b):
        edge_color.append("#FFF580")
    elif ('Monica' in a) or ('Monica' in b):
        edge_color.append("#FF4238")
    elif ('Rachel' in a) or ('Rachel' in b):
        edge_color.append("#FFDC00")
    elif ('Phoebe' in a) or ('Phoebe' in b):
        edge_color.append("#42A2D6")
    elif ('Ross' in a) or ('Ross' in b):
        edge_color.append("#00009E")
    elif ('Joey' in a) or ('Joey' in b):
        edge_color.append("#9A0006")
    else:
        edge_color.append("#9B59B6")

forceatlas2 = ForceAtlas2(outboundAttractionDistribution=False,edgeWeightInfluence=1.5,jitterTolerance=0.1,
barnesHutOptimize=True,barnesHutTheta=1,scalingRatio=1.,strongGravityMode=False,gravity=0.1,verbose=True)
gf = figure(figsize=(20, 20), dpi=320)

positions = forceatlas2.forceatlas2_networkx_layout(G,pos=None,iterations=200)

nx.draw_networkx_nodes(GCC,positions,node_color=colors,node_size=[v*10 for v in dict(GCC.degree()).values()])
nx.draw_networkx_edges(GCC,positions,edge_color=edge_color)

st.pyplot(gf)