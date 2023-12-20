"""Streamlit app."""

import streamlit as st

from edo.cookiebaker.app._utils.server_setup import server_setup
from edo.cookiebaker.app._utils.theme import theme_css
from edo.cookiebaker.predict import get_prediction

@st.cache_data
def get_prediction_(reference_file, file):
    return get_prediction_(reference_file=reference_file, file=file)

def app():
    # page_title = 'Homepage'
    st.set_page_config(page_title='page_title', page_icon=":cookie:", layout="wide")
    st.markdown(theme_css, unsafe_allow_html=True)
    # st.image('./img/logo.png')
    filename = 'tmp/evaluate_reference.py'
    with open(filename) as f:
        reference_file = f.read()

    filename = 'tmp/evaluate.py'
    with open(filename) as f:
        file = f.read()

    with st.spinner('Set up...'):
        server_setup()

    with st.expander("Reference file"):
        st.code(reference_file)
    # st.header("File")
    with st.expander("File"):
        st.code(file)

    p = get_prediction_(reference_file=reference_file, file=file)
    # st.header("File baked")
    with st.expander("File baked"):
        st.code(p)


app()
