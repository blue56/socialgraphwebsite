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

st.balloons()
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
# Giant connected
largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc)


with st.container():
    col1, col2 = st.columns(2)

    with col1:

        col1.header('Welcome to the project website for the course Social graphs and interactions (02805)')
        col1.subheader('Made by Mihaela, Viktor, and Jacob')

    with col2:
        st.pyplot(logic.generate_graph(GCC))
        st.markdown(
            'In the visualization above we can observe the Friends Universe in the shape of a network. The nodes representing the main six character have been colored with distinct colors. '
            'Moreover the node size has been adjusted to be proportional with the degree while the edges have inherited the color of the starting node')



with st.container():


    # === Video
    st.header('Introduction video - The One With the Social Graphs')
    #st.video('https://youtu.be/_qKCQAbOt_8')

    col1, col2 = st.columns(2)

    with col1:
        st.video('https://youtu.be/_qKCQAbOt_8')
    st.balloons()

    # with col2:
    #    col2.header('The One With the Social Graphs')
    #    st.pyplot(logic.generate_graph(GCC))

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
    t = 'There are 194 female characters and 194 male characters in the show. For the rest of 35 characters the gender is unknown. '
    st.markdown(t)
"""
We can easily identify the 6 main characters just by looking at the network degree distribution.
The 6 friends have the highest node degrees.
\nIn other words, they simply have the highest number of relationships to other characters.
That is expected for something that is human-made. The statistics would look differently if the network was totally random.

"""


##################################################################### Degree distribution############################################################################################################################
with st.container():

    st.subheader('Degree Distribution')
    logic.generate_degree_distribution_plot(G)

##################################################################### PLOTLY PLOT #####################################################################################################################################
#friends_links = pd.read_csv('friends_links.csv')
#with st.container():

    #st.header('Degree distributions')
 #   logic.generate_graph(df_nodes,friends_links)


##############################################################################WORDCLOUDS ################################################################################################################################

# Read words for characters from CSV
wordlistPath = "characterwords.csv"
df_wordlist = pd.read_csv(wordlistPath)

st.header('Word cloud drawings')


st.write('But what is happeing at the center of Friends? They are of course talking a lot in the Central Perk café.')
st.write('To be precise, the Friends series has 46657 story lines in total. Each of the 6 friends has their distinct way of talking.')
st.write('Below you can choose your favorite Friends character and see what words make them special.')


with st.container():
    ch_selected = st.selectbox('See the word cloud for:', list(df_wordlist.Name.unique()))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(logic.generate_wordcloud(ch_selected,df_wordlist))
    
    with col2:
        if (ch_selected == "Ross" or ch_selected == "Joey"):
            """
            What do Ross and Joey have in common? _Dude, dude, dude._ Ross and Joey love to say _dude_. They say _dude_ 62 times in total.
            """
        if (ch_selected == "Chandler"):
            """
            Why does Chandler say **_moustache_** so often? Maybe he is still a little jealous of Richard - Monica's only other big love... and his moustache.

            We also see the words **_crystal duck_** from Season 1 Episode 24: _The One Where Rachel Finds Out_.
            
            That was when Chandler blurted out that Ross bought an expensive gift for Rachel just like he did when he was *in love* with Carol in college.
            And Rachel finally _found out_ that Ross was in love with her.

            Is it a coincidence that both words from such a key moment in Ross and Rachel's relationship show up?
            
            Nah, it's just a **_Joincidence_** with a **_C_**.
            """
        elif (ch_selected == "Joey"):
            """            
            We all know Joey and his **_agent Estelle_** have a unique relationship.
            
            Estelle was the person who gave Joey all the opportunities to go to _castings_ and she never had a problem that Joey doesn't quite tell the truth in his **_resume_**.
            
            The things that have a special place in Joey's heart are also here.
            
            **_Hugsy_** - Joey's favorite stuffed animal.
            
            **_The Mets_** - Joey's favorite baseball team.
            
            And **_Dr. Drake Ramoray_** - Joey's iconic character in the famous soap opera - _Days of Our Lives_.
            """
        elif (ch_selected == "Monica"):
            """
            Monica is one of the kindest friends - she always says **_sweetie_** and **_honey_**, and never forgets to ask whether **_Rach_** is okay.

            We do know, however, that Monica has a high-maintenance side. That is also clearly shown by words like **_notes_**, **_ideal_**, and **_efficient_**. 

            And last but not least...
            Damn all the **_JELLYFISH_**!
            """
        elif (ch_selected == "Phoebe"):
            """
            So why is _Minsk_ a special word for Phoebe? Well, that's because her first love, David
            the Scientist Guy, had to move to Minsk for work.

            No wonder Phoebe found David interesting...
            David is actually quite a positive guy, as you can see at the bottom of the page. He has a sentiment score of 0.133, which is above the average.

            Phoebe's twin sister **_Ursula_** appears many times over the seasons. She clearly has a significant role in Pheobe's life.

            Phoebe is a masseuse, which is probably why the word **_massage_** is one of her most used words.

            Hmm... why does the word **_garlic_** appear?
            I bet it is because of time Phoebe and Monica were fighting, and Phoebe said Monica puts **_garlic_** in everything she cooks.
            """
        elif (ch_selected == "Ross"):
            """
            Ross text
            """
        elif (ch_selected == "Rachel"):
            """
            Rachel text
            """

# Gigant connected
largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc)

###################################################################################################### PLOT NETWORK #########################################################################################################
with st.container():
    st.header('Interactive Visualizations of the Friends Universe')

    st.markdown(f'Let\'s look at two interactive graphs of the Friends network. The first one is colored by gender - \
                  <span style="background-color:#03DAC6; color:black">male</span>\
                  , <span style="background-color:#6200EE; color:white">female</span>\
                  , or <span style="background-color:#FFF176; color:black">unknown</span>\
                  . The node size based on the character\'s number of lines.', unsafe_allow_html=True)
    st.markdown('It is know that Friends has an equal number of male and female characters - if we don\'t count the characters whose gender is unknown. The gender distribution is reflected in the graph as well')
    pv_static(logic.generatePyvisGraphGender(df_nodes_attr, df_edges, G, GCC))
    pv_static(logic.generatePyvisGraph(df_nodes, df_edges, G))

    st.markdown(f'On the second graph the main characters are colored - \
                  <span style="background-color:#FFF580; color:black">Chandler</span>\
                  , <span style="background-color:#FF4238; color:white">Monica</span>\
                  , <span style="background-color:#FFDC00; color:black">Rachel</span>\
                  , <span style="background-color:#42A2D6; color:black">Phoebe</span>\
                  , <span style="background-color:#00009E; color:white">Ross</span>\
                  , and <span style="background-color:#9A0006; color:white">Joey</span>', unsafe_allow_html=True)
    st.markdown('Here we have decided to focus again on the main characters and their relationship. Therefore, we have coded the edges to take on the color of their starting node.Can you guess which'
                'main character has formed the most connections? We\'ll dig into that later.')


"""
Let’s go back to Phoebe and her relationships. One of the best relationship in the Friends universe is Phoebe and Mike’s. To be more precise, Mike has the most positive sentiment when talking about Phoebe. This can be seen in the Sentiment for a pair of characters. They have a sentiment score of 0.17 - far above the average.

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


#######################################################################################Centrality###################################################################################################################
characters_sentiment = pd.read_csv('characters_sentiment.csv')

deg_centrality_df = pd.read_csv('top_deg_cent.csv')
betw_centr_df = pd.read_csv('top_betw_centralities.csv')

with st.container():
    st.header('So who is really the most central character?')
    st.markdown('Well according to degree centrality aka who has the highest shares of connections in the network it is Ross!')
    st.markdown('And according to betweenness centrality is...Ross again! Unbelievable. Well we do know that Ross has a rich professional circle as well.')
    col1, col2 = st.columns(2)

    with col1:

        col1.subheader('Degree centrality')
        logic.create_centrality_graphs(deg_centrality_df[:20])
    with col2:

        col2.subheader('Betweenness Centrality')
        logic.create_centrality_graphs_v2(betw_centr_df[:20])


df_lines_words = pd.read_csv("lines_and_words_agg.csv")


with st.container():

    st.markdown(
        'Now let\'s look at who is the character with the most lines ever said in the show. That appears to be Rachel with over 10K sentences and 50K words! Well that can tell us a bit about'
        'who the writers were favoring, does it not ?')

    with col1:
        col1.subheader('Number of lines - spoken in the show')
        option = col1.selectbox('Show me the number of lines for:', ['Secondary Characters', 'Main Characters'])
        if option == 'Secondary Characters':
            logic.generate_bar_chart_sentences(df_lines_words,True)
        elif option == 'Main Characters':
            logic.generate_bar_chart_sentences(df_lines_words, False)


    with col2:
        col2.subheader('Number of sentences spoken in the show')
        option = col2.selectbox('Show me the number of sentences for:', ['Secondary Characters', 'Main Characters'])
        if option == 'Secondary Characters':
            logic.generate_bar_chart_words(df_lines_words, True)
        elif option == 'Main Characters':
            logic.generate_bar_chart_words(df_lines_words, False)

st.markdown('We hope that this visual analysis of the Friends show was insightful! :) ')
st.balloons()