import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_distances
import nltk

nltk.download('stopwords')
nltk.download('punkt') 

if 'model' not in st.session_state or 'data' not in st.session_state or 'matrix' not in st.session_state:
    model = pickle.load(open('model/tf.pkl', 'rb'))
    loaded_sparse_matrix = pickle.load(open('model/bow.pkl', 'rb'))
    st.session_state['model'] = model
    st.session_state['matrix'] = loaded_sparse_matrix
    df = pd.read_csv('dataset/games_steam_clean.csv')
    st.session_state['data'] = df

st.title('Top Game Picks of 2023!')
search, image = st.columns([3,1], gap='large')

with search:
    game = st.selectbox(
        'Which game from 2023 have you played?',
        st.session_state['data']['Name'].tolist()
    )

with image:
    image_link = st.session_state['data'][st.session_state['data']['Name'] == game]['image_link'].to_list()[0]
    title_text = game
    st.image(image_link, width=70)

if st.button('Recommend Me A Fun and Similar Games'):
    st.subheader('Games you\'ll likes :heart:')
    idx = st.session_state['data'][st.session_state['data']['Name'] == game].index[0]
    content = st.session_state['data'].loc[idx, 'metadata']
    watched = st.session_state['model'].transform([content])
    dist = cosine_distances(watched, st.session_state['matrix'])
    rec_idx = dist.argsort()[0, 1:11]
    col0, col1, col2, col3, col4 = st.columns(5)
    col5, col6, col7, col8, col9 = st.columns(5)
    list_col = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9]
    for idx, val in enumerate(list_col):
        link = st.session_state['data'].loc[rec_idx].iloc[idx]['image_link']
        title = st.session_state['data'].loc[rec_idx].iloc[idx]['Name']
        with val:
            st.image(link, caption=title, width=150)

    st.write('Dataset Detail')
    st.dataframe(st.session_state['data'].loc[rec_idx][['Name', 'metadata']])
else:
    st.write('Waiting')
