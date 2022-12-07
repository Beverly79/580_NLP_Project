#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pyhere import here
from numpy import dot
from numpy.linalg import norm
import streamlit as st
import time


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


pd.options.mode.chained_assignment = None


def recommendation(article_index):
    print("# Input Healine: ", df_ner.iloc[article_index]["headline"])
    print("# Input Label: ", df_ner.iloc[article_index]["label"])
    similarity_dict = {}
    for i in range(arr_embedding.shape[0]):
        similarity_dict[i] = cosine_similarity(
            arr_embedding[article_index], arr_embedding[i]
        )
    similarity_ranks = list(
        sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
    )
    similarity_ranks = [i for i in similarity_ranks if i[0] != article_index]
    top10_similar_articles = similarity_ranks[:10]

    index_list = [index for index, score in top10_similar_articles]
    score_list = [score for index, score in top10_similar_articles]
    df1 = df_ner.iloc[index_list]
    df1["cosine_similarity"] = score_list
    print("# Top 10 similar articles")
    return df1[["headline", "text", "cosine_similarity", "ner"]]


df_embedding = pd.read_csv(here("pretrain_embeddings.csv"))
df_embedding = df_embedding.drop("Unnamed: 0", axis=1)

# df_embedding.head()
# df_embedding.shape

df_ner = pd.read_csv(here("entitied_article.csv"))
df_ner = df_ner.drop("Unnamed: 0", axis=1)

# df_ner.head()
# df_ner.shape

arr_embedding = df_embedding.to_numpy()

# arr_embedding.shape
# arr_embedding[0].shape

# cosine_similarity(arr_embedding[0], arr_embedding[1])

st.write(
    """
         # Document Recommender System
         """
)

indices = df_ner.index.values

with st.sidebar:
    st.write("### Select Your Article")
    headline = st.selectbox("Article:", df_ner["text"].tolist())
    article_id = indices[df_ner["text"] == headline]
    st.info(df_ner["text"].iloc[article_id].tolist()[0], icon="ℹ️")


st.write(recommendation(article_id))

