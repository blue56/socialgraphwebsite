import random
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
import streamlit
from fa2 import ForceAtlas2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from plotly.graph_objs import Scatter, Figure
from wordcloud import WordCloud
import plotly.graph_objects as go
from pyvis import network as net
from pyvis import options
import altair as alt

def generate_network(df_nodes,df_edges):

    G = nx.Graph()
    # Add nodes to the graph
    for index, character in df_nodes.iterrows():
        if (character["Name"]):
            G.add_node(character["Name"])


    friends_links = list(zip(df_edges.From, df_edges.To))
    G.add_edges_from(friends_links)

    return G

def generate_degree_distribution_plot(G):

    # get all sentiment values
    degrees = pd.DataFrame()
    degrees['degree'] = [i[1] for i in list(G.degree)]
    # get hist values and edges
    #hist, bin_edges = np.histogram(degrees)

    fig = px.histogram(degrees, x='degree')
    return streamlit.plotly_chart(fig, use_container_width=True)


def generate_wordcloud(champion,df_wordlist):

    colormapList = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #champion = champion_row[1][0]
    wordcloud_string = df_wordlist[df_wordlist.Name == champion]['Words'].values[0]

    # Create and generate a word cloud image:
    colormap = random.choice(colormapList)
    wordcloud = WordCloud(width=600, height=600, background_color='white', colormap=colormap).generate(wordcloud_string)
    wordcloud.collocations = False

    # Display the generated image:
    fig = plt.figure( figsize=(5,5))
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.title(champion,fontsize=15)
    plt.show()

    return fig



def generate_graph(GCC):



    val_map = {'Ross': 1,'Chandler': 2,'Joey': 3,'Phoebe': 4,'Monica':5,'Rachel': 6,'Other': 7}
    #I had this list for the name corresponding t the color but different from the node name
    ColorLegend = {'Ross': 1,'Chandler': 2,'Joey': 3,'Phoebe': 4,'Monica':5,'Rachel': 6,'Other': 7}

    colors = []
    for node in list(GCC.nodes()):
        if node == 'Ross':
            colors.append("#00009E")
        elif node == 'Chandler':
            colors.append("#FFF580")
        elif node == 'Joey':
            colors.append("#9A0006")
        elif node == 'Phoebe':
            colors.append("#42A2D6")
        elif node == 'Rachel':
            colors.append("#FFDC00")
        elif node =='Monica':
            colors.append("#FF4238")
        else:
            colors.append("#9B59B6")


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

    #legend_colors = ["#FFF580","#FF4238","#FFDC00","#42A2D6","#00009E","#9A0006"]

    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=False,edgeWeightInfluence=1.5,jitterTolerance=0.1,
    barnesHutOptimize=True,barnesHutTheta=1,scalingRatio=1.,strongGravityMode=False,gravity=0.1,verbose=True)
    gf = figure(figsize=(40, 40), dpi=320)

    ax = gf.add_subplot(1,1,1)
    for label in ColorLegend:
        legend_color = ""
        if label == "Ross":
            legend_color = "#00009E"
        elif label == "Chandler":
            legend_color = "#FFF580"
        elif label == "Joey":
            legend_color = "#9A0006"
        elif label == "Phoebe":
            legend_color = "#42A2D6"
        elif label == "Rachel":
            legend_color = "#FFDC00"
        elif label == "Monica":
            legend_color = "#FF4238"
        else:
            legend_color = "#9B59B6"

        ax.plot([0],[0],color=legend_color,label=label)

    positions = forceatlas2.forceatlas2_networkx_layout(GCC,pos=None,iterations=200)

    nx.draw_networkx_nodes(GCC,positions,node_color=colors,node_size=[v*10 for v in dict(GCC.degree()).values()])
    nx.draw_networkx_edges(GCC,positions,edge_color=edge_color)

    plt.legend(prop={'size': 30})

    gf.tight_layout()

    return gf

def create_sentiment_graph(sentiment_df):

    if sentiment_df.columns[1] == 'Directors':
        fig = px.bar(sentiment_df, x=sentiment_df.columns[1], y=sentiment_df.columns[2])


    elif sentiment_df.columns[1] == 'Episodes':
        fig = px.line(sentiment_df, x=sentiment_df.columns[1], y=sentiment_df.columns[2])

    elif sentiment_df.columns[1] == 'Character':

        sentiment_df['colors'] = ''
        sentiment_df['colors'] = np.where(sentiment_df['Sentiment'] <= -0.05, 'negative', sentiment_df['colors'])
        sentiment_df['colors'] = np.where(sentiment_df['Sentiment'] >= 0.05, 'positive', sentiment_df['colors'])
        sentiment_df['colors'] = np.where( ((sentiment_df['Sentiment'] > -0.05) & (sentiment_df['Sentiment'] < 0.05 )), 'neutral', sentiment_df['colors'])

        fig = px.scatter(sentiment_df, x=sentiment_df.columns[1], y=sentiment_df.columns[2], color='colors')
        for i in range(0,len(sentiment_df['Character'])):
            if sentiment_df['Character'].iloc[i] in ['Monica', 'Rachel', 'Ross', 'Chandler', 'Joey','Phoebe']:
                fig.add_annotation(x = sentiment_df.Character.iloc[i], y = sentiment_df.Sentiment.iloc[i],arrowhead=1)
                #, xref = "x domain", yref = "y", axref = "x domain", ayref = "y", ay = 2, arrowhead = 2)

    else:
        fig = px.bar(sentiment_df, x=sentiment_df.columns[1], y=sentiment_df.columns[2])

    return streamlit.plotly_chart(fig, use_container_width=True)

def generatePyvisGraph(df_nodes, df_edges, G):
    # Create pyvis graph
    g = net.Network(height='700px', width='1200px', notebook=False, heading='', bgcolor='#00000', font_color='white')
    g.barnes_hut(gravity=-80000, central_gravity=0, overlap=1)
    g.set_edge_smooth('continuous')

    # Add colors for main character nodes
    nodes_to_remove = []
    for node in G:
        if node == 'Chandler':
            g.add_node('Chandler', label='Chandler', color="#FFF580")
            nodes_to_remove.append(node)
        elif node == 'Monica':
            g.add_node('Monica', label='Monica', color="#FF4238")
            nodes_to_remove.append(node)
        elif node == 'Rachel':
            g.add_node('Rachel', label='Rachel', color="#FFDC00")
            nodes_to_remove.append(node)
        elif node == 'Phoebe':
            g.add_node('Phoebe', label='Phoebe', color="#42A2D6")
            nodes_to_remove.append(node)
        elif node == 'Ross':
            g.add_node('Ross', label='Ross', color="#00009E")
            nodes_to_remove.append(node)
        elif node == 'Joey':
            g.add_node('Joey', label='Joey', color="#9A0006")
            nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)

    friends_links = list(zip(df_edges.From, df_edges.To))
    G.add_edges_from(friends_links)

    # Gigant connected 
    largest_cc = max(nx.connected_components(G), key=len)
    GCC = G.subgraph(largest_cc)

    # Add default colored nodes to pyvis graph
    g.add_nodes(GCC.nodes)

    # Add color for main character edges
    edges_to_remove = []
    for edge in list(G.edges()):
        a = edge[0]
        b = edge[1]
        if ('Chandler' in a) or ('Chandler' in b):
            g.add_edge(a, b, color = "FFF580")
            edges_to_remove.append(edge)
        elif ('Monica' in a) or ('Monica' in b):
            g.add_edge(a, b, color = "FF4238")
            edges_to_remove.append(edge)
        elif ('Rachel' in a) or ('Rachel' in b):
            g.add_edge(a, b, color = "FFDC00")
            edges_to_remove.append(edge)
        elif ('Phoebe' in a) or ('Phoebe' in b):
            g.add_edge(a, b, color = "42A2D6")
            edges_to_remove.append(edge)
        elif ('Ross' in a) or ('Ross' in b):
            g.add_edge(a, b, color = "00009E")
            edges_to_remove.append(edge)
        elif ('Joey' in a) or ('Joey' in b):
            g.add_edge(a, b, color = "9A0006")
            edges_to_remove.append(edge)

    # Add default colored edges to pyvis graph
    G.remove_edges_from(edges_to_remove)
    g.add_edges(G.edges)
    return g

def generatePyvisGraphGender(df_nodes, df_edges, G, GCC):
    # Create pyvis graph
    g = net.Network(height='700px', width='1200px', notebook=False, heading='', bgcolor='#00000', font_color='white')
    g.barnes_hut(gravity=-80000, central_gravity=0, overlap=1)
    g.set_edge_smooth('continuous')

    # Get df_names if existing in GCC
    df_nodes = df_nodes[df_nodes.Name.isin(list(GCC.nodes))]

    names = df_nodes['Name'].values
    num_lines = 7 * np.log([1 if val == 0 else val for val in df_nodes['No_of_Lines'].values])
    genders = df_nodes['Gender'].values

    # Get node colors based on gender
    node_colors = ['#03DAC6' if gender == 'male' else '#6200EE' if gender == 'female' else '#FFF176' for gender in genders]
    
    # Add nodes with color based on gender and size based on number of lines
    for i in range(len(names)):
        g.add_node(names[i], label=names[i], color=node_colors[i], size=num_lines[i])

    g.add_edges(G.edges)
    return g


import plotly.graph_objects as go
def create_centrality_graphs(df_centrality):

    colors = ['#ff7f0e', ] * len(df_centrality)
    fig = go.Figure(data=[go.Bar(x=list(df_centrality['Character']), y=list(df_centrality['Value']),marker_color=colors)])
    #fig = px.bar(df_centrality, x=df_centrality.columns[1], y=df_centrality.columns[2], color = colors)

    return streamlit.plotly_chart(fig, use_container_width=True)

def create_centrality_graphs_v2(df_centrality):

    colors = ['#bcbd22', ] * len(df_centrality)
    fig = go.Figure(data=[go.Bar(x=list(df_centrality['Character']), y=list(df_centrality['Value']),marker_color=colors)])

    return streamlit.plotly_chart(fig, use_container_width=True)

def generate_bar_chart(df_lines_words):
    df_lines_words = df_lines_words[["Character", "no_sentences", "no_words"]]
    df_lines_words = df_lines_words.loc[df_lines_words['no_sentences'] > 93]

    return alt.Chart(df_lines_words).mark_bar(opacity=0.9).encode(
        x='no_sentences',
        y=alt.Y('Character', sort='-x'),
        tooltip=['Character', 'no_sentences', 'no_words']
    )
