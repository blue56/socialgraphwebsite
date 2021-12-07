import random
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
import streamlit
from fa2 import ForceAtlas2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from wordcloud import WordCloud


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
    degrees = [i[1] for i in list(G.degree)]
    # get hist values and edges
    hist, bin_edges = np.histogram(degrees)

    fig = figure(figsize=(12, 10), dpi=80)
    n, bins, patches = plt.hist(x=degrees, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Degree', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title('$\it{F}$•$\it{R}$•$\it{I}$•$\it{E}$•$\it{N}$•$\it{D}$•$\it{S}$ - Degree Distribution', fontsize=18)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.xticks(np.arange(min(degrees), max(degrees) + 1, 10))
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    return fig

def generate_wordcloud(champion,df_wordlist):

    colormapList = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #champion = champion_row[1][0]
    wordcloud_string = df_wordlist[df_wordlist.Name == champion]['Words'].values[0]

    # Create and generate a word cloud image:
    colormap = random.choice(colormapList)
    wordcloud = WordCloud(width=1000, height=1000, background_color='white', colormap=colormap).generate(wordcloud_string)
    wordcloud.collocations = False

    # Display the generated image:
    fig = plt.figure( figsize=(5,10))
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
    gf = figure(figsize=(20, 20), dpi=320)

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


