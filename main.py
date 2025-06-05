import streamlit as st
import time 

st.title("Programmer Sarswat")
st.subheader("Select and upload file-")

file = st.file_uploader("", type=["pdf", "xls", "xlsx"])
upload_clicked = st.button("Upload File")
def add():
    st.title("🤖 Programmer Sarswat")

    st.chat_input("find anything from database.")


if file and upload_clicked:
    st.success(f"You have uploaded file `{file.name}` successfully.")
    with st.spinner("Redirecting please wait..."):
        time.sleep(2)
        add()  # Call the add function to show the AI chatbot interface

elif upload_clicked and not file:
    st.warning("⚠️ Please upload a file.")




































# import streamlit as st
# import time
# import openai

# st.title("Programmer Sarswat")
# st.subheader("Select and upload file-")

# file = st.file_uploader("", type=["pdf", "xls", "xlsx"])
# a = st.button("Upload File")

# if file and a:
#     st.success(f"You have uploaded file `{file.name}` successfully.")
#     with st.spinner("Redirecting please wait..."):
#         time.sleep(2)
        
# elif a and not file:
#     st.warning("⚠️ Please upload a file.");


    
