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
from stvis import pv_static
st.set_page_config(page_title="Friends • The Network",layout='wide')

# ===

st.header('Welcome to the project website for the course Social graphs and interactions (02805)')
st.subheader('Made by Mihaela-Elena Nistor, Viktor Anzhelev Tsanev and Jacob Kofod')

################################################################## Read characters from CSV###################################################################################################################
characters_path = "nodes.csv"
df_characters = pd.read_csv(characters_path)

path_nodes = "nodes.csv"
df_nodes = pd.read_csv(path_nodes)
path_nodes_attr = "nodes_with_attributes.csv"
df_nodes_attr = pd.read_csv(path_nodes_attr)
path_edges = "edges.csv"
df_edges = pd.read_csv(path_edges)
##################################################################################CREATE NETWORK#################################################################################################################
G = logic.generate_network(df_nodes, df_edges)
# Gigant connected
largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc)

with st.container():


    # === Video
    #st.header('Introduction video')
    #st.video('https://youtu.be/_qKCQAbOt_8')

    col1, col2 = st.columns(2)

    with col1:
        col1.header('Introduction Video')
        st.video('https://youtu.be/_qKCQAbOt_8')

    with col2:
        col2.header('The Friends Universe')
        st.pyplot(logic.generate_graph(GCC))

###################################################################################### Basic stats###############################################################################################################


st.header('Introduction')
"""
Friends is all about relationships, but what is actually happening at Central Perk? What are the friends talking about? Who does actually have the best relationship in Friends? Let's start to dig into the Friends network and let’s see what we can find from a data perspective.

First of all, we need to have the data available. We have done the hard work for you, so this website has all the needed information available. However, if you wish to jump into the data lake yourself then you can find the details here:

[Link to explainer notebook]

Let's take a look at the character network. It consists of 424 characters that are connected by 2183 edges. Each edge represents that the character has some kind of a relationship to the other character.

"""

with st.container():


    st.header('Basic stats')

    numberOfNodes = len(G.nodes())
    numberOfEdges = len(G.edges())

    t = "The network has " + str(numberOfNodes) + " nodes and " + str(numberOfEdges) + " edges."
    st.markdown(t)

"""
We can identify the 6 main characters just by looking at the network degree distribution.
The 6 friends have the highest node degrees.
\nIn other words, they simply have the highest number of relationships to other characters.
That is expectable for something that is human-made. The statistics would look differently if the network was totally random.

"""

##################################################################### Degree distribution############################################################################################################################
with st.container():

    st.header('Degree distributions')
    logic.generate_degree_distribution_plot(G)

##################################################################### PLOTLY PLOT #####################################################################################################################################
friends_links = pd.read_csv('friends_links.csv')
with st.container():

    st.header('Degree distributions')
    logic.generate_plotly_graph(df_nodes,friends_links)


##############################################################################WORDCLOUDS ################################################################################################################################

# Read words for characters from CSV
wordlistPath = "characterwords.csv"
df_wordlist = pd.read_csv(wordlistPath)

st.header('Word cloud drawings')



st.write('But what is happing at the center of Friends? They are of course talking a lot in the Central Perk café.')
st.write('To be precise, the Friends series has 46657 story lines in total. Each of the 6 friends has their distinct way of talking.')
st.write('Below you can choose your favorite Friends character and see what words make them special.')


with st.container():
    ch_selected = st.selectbox('See the word cloud for:', list(df_wordlist.Name.unique()))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(logic.generate_wordcloud(ch_selected,df_wordlist))
    
    with col2:
        """
        So why is _Minsk_ a special word for Pheobe? Well, that's because her first love, David
        the Scientist Guy, had to move to Minsk for work.

        No wonder Phoebe found David interesting...
        David is actually quite a positive guy, as you can see at the bottom of the page. He has a sentiment score of 0.133, which is above the average.

        What do Ross and Joey have in common? _Dude, dude, dude._ Ross and Joey love to say _dude_. They say _dude_ 62 times in total.
        """

# Gigant connected
largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc)

###################################################################################################### PLOT NETWORK #########################################################################################################
with st.container():

    st.header('Interactive Friends Graphs')
    # st.pyplot(logic.generate_graph(GCC))
    
    st.markdown(f'Let\'s look at two interactive graphs of the Friends network. The first one is colored by gender - \
                  <span style="background-color:#03DAC6; color:black">male</span>\
                  , <span style="background-color:#6200EE; color:white">female</span>\
                  , or <span style="background-color:#FFF176; color:black">unknown</span>\
                  . The node size based on the character\'s number of lines.', unsafe_allow_html=True)

    pv_static(logic.generatePyvisGraphGender(df_nodes_attr, df_edges, G, GCC))

    st.markdown(f'On the second graph the main characters are colored - \
                  <span style="background-color:#FFF580; color:black">Chandler</span>\
                  , <span style="background-color:#FF4238; color:white">Monica</span>\
                  , <span style="background-color:#FFDC00; color:black">Rachel</span>\
                  , <span style="background-color:#42A2D6; color:black">Phoebe</span>\
                  , <span style="background-color:#00009E; color:white">Ross</span>\
                  , and <span style="background-color:#9A0006; color:white">Joey</span>', unsafe_allow_html=True)
    
    pv_static(logic.generatePyvisGraph(df_nodes, df_edges, G))

"""
Let’s go back to Phoebe and her relationships. It seems that the best relationship in the Friends universe is Phoebe and Mike’s. To be more precise, Mike has the most positive sentiment when talking about Phoebe. This can be seen in the Sentiment for a pair of characters. They have a sentiment score of 0.17 - far above the average.

But wait a bit... Mike was dating Precious after he broke up with Phoebe. That must have been a challenge. Precious has the most negative sentiment of all characters in Friends. Ouch! Mike must have been on a roller coaster ride.

Speaking about roller coaster rides… The Friends viewers have also been on a bit of a ride.
The episode with the lowest overall sentiment was Episode 1 in Season 4: _The one with the Jellyfish_, as you probably remember.
The crazy thing is that in the same season we have one of the highest sentiment scores of an episode - Episode 5: _The one with Joey's new girlfriend_.
"""

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
#######################################################################################SENTIMENT CHARACTERS###################################################################################################################
characters_sentiment = pd.read_csv('characters_sentiment.csv')

with st.container():
    logic.create_sentiment_graph(characters_sentiment)

df_lines_words = pd.read_csv("lines_and_words_agg.csv")
with st.container():
    st.altair_chart(logic.generate_bar_chart(df_lines_words), use_container_width=True)
