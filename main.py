import streamlit as st
import time 

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False


st.title("PDF Chatbot")

if not st.session_state.file_uploaded:

    st.subheader("Select and upload file-")

    file = st.file_uploader("", type=["pdf"])
    upload_clicked = st.button("Upload File")
    if file and upload_clicked:
        st.success(f"You have uploaded file `{file.name}` successfully.")
        st.session_state.file_uploaded = True
        with st.spinner("Redirecting please wait..."):
            time.sleep(2)
            st.rerun()
        
    elif upload_clicked and not file:
        st.warning("⚠️ Please upload a file.")



if st.session_state.file_uploaded:
    st.success("File uploaded successfully! Now you can interact with the PDF.")
    # Here you can add the code to process the uploaded PDF and create a chatbot interface
    # For example, you could use PyPDF2 or pdfminer to extract text from the PDF and then use OpenAI's API to create a chatbot.
    
    # Example placeholder for chatbot interaction
    st.text_input("Ask a question about the PDF:")
    st.button("Submit")




































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


    
