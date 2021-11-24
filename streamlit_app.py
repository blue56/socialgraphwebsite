from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to the project website for the course Social graphs and interactions (02805)
# Made by Mihaela-Elena Nistor, Viktor Anzhelev Tsanev and Jacob Kofod

"""

# Read characters from CSV
characters_path = "nodes.csv"
df_characters = pd.read_csv(characters_path)

print(df_characters)