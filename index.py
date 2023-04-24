"""
importing all the libraries need 
"""
import os
import pickle
import streamlit as st 
import predict_data as prd

# importing a css file
with open("style.css") as f:
    st.markdown('''<style>{}</style>'''.format(f.read()), unsafe_allow_html=True)

CATEGORI=list(pickle.load(open("categories.pkl","rb")))
CATEGORI.insert(0,"All")

PATH="test_resume"
# Title of the web app
st.title("Resume Recommendation System")

# creat a form 
with st.form(key='my_form'):
    option = st.selectbox(
    'Enter a keyword : ',
    tuple(CATEGORI))
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        outputs,uid=prd.test_data(PATH)
        if option =="All":
            output=prd.filter_data(outputs,uid)
        else:
            output=prd.filter_data(outputs,uid,option)
        if output is not None:
                st.dataframe(output)
        else:
            st.write("No Match found!")

with st.form(key="my_form1"):
    file=st.file_uploader("Upload the resume here..",accept_multiple_files=False,type=["pdf"])

    submitted=st.form_submit_button(label="upload")

    if submitted:
        try:
            os.mkdir("temp")
        except:
            pass
        with open("temp/uploaded.pdf", "wb") as f:
            f.write(file.getvalue())
        outputs,uid=prd.test_data("temp")
        output=prd.filter_data(outputs,uid)
        st.dataframe(output)
        