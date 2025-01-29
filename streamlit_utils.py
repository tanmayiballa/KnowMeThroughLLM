import streamlit as st

def sidebar():
    LinkedIn_url = 'https://www.linkedin.com/in/tanmayiballa/'
    st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Tanmayi Balla</h1>", unsafe_allow_html=True)
    st.sidebar.image("my-avatar-2.jpeg", use_column_width=True)
    st.sidebar.markdown("<p style='text-align: center;'><strong>Data Scientist @ UCLA </strong></p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: blue'><strong>tanmayiballa@gmail.com || +1 812-778-4651 </strong></p>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='text-align: center;'><a href='https://www.linkedin.com/in/tanmayiballa'>LinkedIn</a> | <a href='https://tanmayiballa.github.io/'>Portfolio</a> | <a href='https://scholar.google.com/citations?user=bZeCWlwAAAAJ&hl=en&oi=ao'>Google Scholar</a></div>",
        unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'><strong>Actively seeking full-time opportunities. Open to relocation </strong></p>", unsafe_allow_html=True)
