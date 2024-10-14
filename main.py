import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from urllib.parse import quote
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Firebase configuration
firebase_config = {
    "apiKey": "AIzaSyAVYykWmf8B9HPCcSFz6tZugNC5eS7vefQ",
    "authDomain": "prithvi-45d3f.firebaseapp.com",
    "projectId": "prithvi-45d3f",
    "storageBucket": "prithvi-45d3f.appspot.com",
    "messagingSenderId": "292756309891",
    "appId": "1:292756309891:web:56094a7f63f1ee0d88d8b6",
    "databaseURL": "https://prithvi-45d3f-default-rtdb.firebaseio.com"
}

# Initialize Firebase for authentication
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Initialize Firebase Admin for database access
cred = credentials.Certificate('prithvi-45d3f-firebase-adminsdk-o4c77-62e1d077aa.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Custom CSS for styling the app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/free-vector/blue-gradient-blank-background-business_53876-120508.jpg');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def display_error_message(e):
    error_message = str(e)
    if "WEAK_PASSWORD" in error_message:
        st.error("Weak password. It should be at least 6 characters.")
    elif "EMAIL_EXISTS" in error_message:
        st.error("This email is already registered. Try logging in.")
    elif "INVALID_EMAIL" in error_message:
        st.error("Invalid email address. Please check your input.")
    elif "EMAIL_NOT_FOUND" in error_message:
        st.error("No account found with this email. Please sign up first.")
    elif "INVALID_PASSWORD" in error_message:
        st.error("Incorrect password. Please try again.")
    else:
        st.error("Press login again and wait...")

def login_signup():
    st.title("Welcome to DecisionLens🧠")
    st.write("Kindly use Light mode, from settings -> right top corner for better ui/ux")
    st.subheader("Login / Sign Up ")

    choice = st.selectbox("Select an option:", ["Login", "Sign Up"], index=0)

    email = st.text_input("Enter your email ✉️", placeholder="you@example.com")
    password = st.text_input("Enter your password 🔑", type="password", placeholder="Your Password")

    if choice == "Sign Up":
        if st.button("Sign Up 📝"):
            try:
                user = auth.create_user_with_email_and_password(email, password)
                st.success("Account created successfully! Please log in.")
            except Exception as e:
                display_error_message(e)

    elif choice == "Login":
        if st.button("Login 🚪"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.success("Login successful! 🌟")
                st.balloons()
                st.session_state.user = user
                st.session_state.email = email
                st.experimental_rerun()
            except Exception as e:
                display_error_message(e)

def landing_page():
    st.title("DecisionLens: AI-Powered Business Strategy Assistant")
    st.write(f"Welcome {st.session_state.email} 🌼")
    st.write("DecisionLens is your AI-powered companion for data-driven business strategies.")
    
    feature = st.selectbox("Select a feature:", [
        "Market Analysis",
        "Business Strategy Assistant",
        "Data Strategy Simulator"
    ])

    if feature == "Market Analysis":
        market_analysis()
    elif feature == "Business Strategy Assistant":
        business_strategy_assistant()
    elif feature == "Data Strategy Simulator":
        data_strategy_simulator()

def market_analysis():
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    from io import StringIO
    from langchain_groq import ChatGroq
    from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    import tempfile
    import os


    # Custom CSS for the chat box always visible
    st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #4CAF50;
            border-radius: 5px;
        }
        .chat-input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            padding: 10px;
            background-color: #f9f9f9;
            border-top: 1px solid #cccccc;
        }
    </style>
    """, unsafe_allow_html=True)

    # API keys
    ALPHA_VANTAGE_API_KEY = "60HGRBWV0CRP2Q24"
    GROQ_API_KEY = "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"

    # Initialize Groq LLM
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.3)

    # Sample data for commodities
    def get_commodities():
        return pd.DataFrame({
            'Symbol': ['CL', 'BZ', 'NG', 'HG', 'AL', 'W', 'C', 'CT', 'SB', 'KC', 'GC'],
            'Name': ['Crude Oil (WTI)', 'Crude Oil (Brent)', 'Natural Gas', 'Copper', 'Aluminum', 'Wheat', 'Corn', 'Cotton', 'Sugar', 'Coffee', 'Gold']
        })

    # Sample data for companies
    def get_top_companies():
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK.A', 'NVDA', 'JPM', 'JNJ',
                    'V', 'PG', 'UNH', 'HD', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM', 'NFLX',
                    'CMCSA', 'XOM', 'VZ', 'COST', 'INTC', 'ABT', 'PFE', 'CSCO', 'TMO', 'ACN',
                    'AVGO', 'PEP', 'NKE', 'ORCL', 'MRK', 'KO', 'WMT', 'T', 'MCD', 'LLY',
                    'DHR', 'NEE', 'ABBV', 'PYPL', 'PM', 'TXN', 'UNP', 'HON', 'QCOM', 'AMD'],
            'Name': ['Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc.', 'Facebook, Inc.', 'Tesla, Inc.', 
                    'Berkshire Hathaway Inc.', 'NVIDIA Corporation', 'JPMorgan Chase & Co.', 'Johnson & Johnson',
                    'Visa Inc.', 'Procter & Gamble Company', 'UnitedHealth Group Incorporated', 'The Home Depot, Inc.', 
                    'Bank of America Corporation', 'Mastercard Incorporated', 'The Walt Disney Company', 'Adobe Inc.', 
                    'Salesforce.com, inc.', 'Netflix, Inc.', 'Comcast Corporation', 'Exxon Mobil Corporation', 
                    'Verizon Communications Inc.', 'Costco Wholesale Corporation', 'Intel Corporation', 
                    'Abbott Laboratories', 'Pfizer Inc.', 'Cisco Systems, Inc.', 'Thermo Fisher Scientific Inc.', 
                    'Accenture plc', 'Broadcom Inc.', 'PepsiCo, Inc.', 'NIKE, Inc.', 'Oracle Corporation', 
                    'Merck & Co., Inc.', 'The Coca-Cola Company', 'Walmart Inc.', 'AT&T Inc.', 'McDonald\'s Corporation', 
                    'Eli Lilly and Company', 'Danaher Corporation', 'NextEra Energy, Inc.', 'AbbVie Inc.', 
                    'PayPal Holdings, Inc.', 'Philip Morris International Inc.', 'Texas Instruments Incorporated', 
                    'Union Pacific Corporation', 'Honeywell International Inc.', 'QUALCOMM Incorporated', 
                    'Advanced Micro Devices, Inc.']
        })

    # Function to fetch data from Alpha Vantage
    def fetch_data(symbol, function):
        base_url = 'https://www.alphavantage.co/query'
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'datatype': 'csv'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            # Display raw data for inspection
            data = pd.read_csv(StringIO(response.text))
            st.write(data.head())  # This helps to check column names

            # Check if the expected 'timestamp' column exists
            if 'timestamp' in data.columns:
                return pd.read_csv(StringIO(response.text), parse_dates=['timestamp'])
            elif 'date' in data.columns:
                return pd.read_csv(StringIO(response.text), parse_dates=['date'])
            else:
                st.error("Date column not found in the data.")
                return None
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return None


    # File uploader function to process uploaded documents
    def process_document(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.type == "text/csv":
                loader = CSVLoader(tmp_file_path)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = Docx2txtLoader(tmp_file_path)
            else:
                raise ValueError("Unsupported file type")
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(texts, embeddings)
            return vector_store
        finally:
            os.unlink(tmp_file_path)

    # Sidebar for input
    st.sidebar.header("Real-time Market Analysis Options")
    options = st.sidebar.radio("Select Analysis Type", ("Commodities", "Companies"))

    if options == "Commodities":
        commodities = get_commodities()
        selected_commodity = st.sidebar.selectbox("Select a commodity", commodities['Name'])
        symbol = commodities.loc[commodities['Name'] == selected_commodity, 'Symbol'].iloc[0]
    else:
        companies = get_top_companies()
        selected_company = st.sidebar.selectbox("Select a company", companies['Name'])
        symbol = companies.loc[companies['Name'] == selected_company, 'Symbol'].iloc[0]

    interval = st.sidebar.selectbox("Select Time Interval", ["daily", "weekly", "monthly"])

    # File uploader for document analysis
    uploaded_file = st.sidebar.file_uploader("Upload a document for analysis", type=["pdf", "csv", "docx"])

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            vector_store = process_document(uploaded_file)
        st.sidebar.success("Document processed successfully!")

    # Chat continuation for queries
    st.subheader("Enter your query for analysis:")
    user_query = st.text_input("Enter your analysis query:")

    if st.button("Analyze"):
        with st.spinner("Analyzing data..."):
            try:
                # Fetch data
                data = fetch_data(symbol, f'TIME_SERIES_{interval.upper()}')
                
                if data is not None:
                    # Create candlestick chart
                    fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],
                                                        open=data['open'],
                                                        high=data['high'],
                                                        low=data['low'],
                                                        close=data['close'])])
                    fig.update_layout(title=f"{symbol} Price Analysis", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Perform analysis using LLM and vector store
                    if 'vector_store' in locals():
                        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
                        response = qa_chain.run(user_query)
                        
                        st.subheader("Analysis Results")
                        st.write(response)
                    else:
                        st.warning("Please upload a document for a detailed analysis.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("This could be due to API limitations or invalid input. Try again with a different selection.")


def business_strategy_assistant():
    import streamlit as st
    from langchain_groq import ChatGroq
    from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.chains.question_answering import load_qa_chain
    import tempfile
    import os



    # Sidebar for API key input
    with st.sidebar:
        st.title("Configuration")
        user_api_key = st.text_input("Enter your Groq API key", type="password")
        st.info("Your API key is required to Chat with our assistant.")

    # Main title and description
    st.title("📈 Business Strategy Assistant")
    st.write("Upload documents and ask questions to get strategic business insights.")

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Check if API key is provided
    if not user_api_key:
        st.warning("Please enter your Groq API key in the sidebar to begin.")
    else:
        # Initialize Groq API with user-provided key
        llm = ChatGroq(groq_api_key=user_api_key, model_name="mixtral-8x7b-32768", temperature=0.7)

        # File upload section for document input
        uploaded_file = st.file_uploader("Upload PDF, CSV, or DOCX", type=["pdf", "csv", "docx"])

        # Function to process document
        def process_document(file):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif file.name.endswith(".csv"):
                loader = CSVLoader(tmp_file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_file_path)
            else:
                st.error("Unsupported file format.")
                return None

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            doc_chunks = text_splitter.split_documents(documents)
            
            os.remove(tmp_file_path)
            return doc_chunks

        # Function to create FAISS index
        def create_faiss_index(doc_chunks):
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(doc_chunks, embeddings)
            return vector_store

        # Process uploaded document
        if uploaded_file:
            with st.spinner("Processing document..."):
                doc_chunks = process_document(uploaded_file)
                if doc_chunks:
                    vector_store = create_faiss_index(doc_chunks)
                    st.success("Document processed successfully!")
                    
                    # Create a QA chain
                    qa_chain = RetrievalQA(
                        combine_documents_chain=load_qa_chain(llm, chain_type="stuff"),
                        retriever=vector_store.as_retriever(),
                    )
                    
                    # Store QA chain in session state
                    st.session_state.qa_chain = qa_chain

        # Chat interface
        st.subheader("Chat")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask about business strategy"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if hasattr(st.session_state, 'qa_chain'):
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Prompt engineering for business strategy focus
                        business_prompt = f"""As a business strategy expert, provide insights and recommendations based on the following question:

                        {prompt}

                        Focus your response on strategic business concepts, market analysis, competitive advantage, and actionable recommendations. Be concise and practical in your advice."""
                        
                        response = st.session_state.qa_chain.run(business_prompt)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Please upload a document first.")

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

def data_strategy_simulator():
    st.title("🎯 AI-Powered Data Strategy Simulator")
    st.write("Connect to your local database(MySql/PostgreSQL/SQLlite,etc)")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    db_config = {
        "host": st.sidebar.text_input("Database Host", value="localhost"),
        "port": st.sidebar.text_input("Database Port", value="3306"),
        "user": st.sidebar.text_input("Database User", value="root"),
        "password": st.sidebar.text_input("Database Password", type="password"),
        "database": st.sidebar.text_input("Database Name"),
    }
    api_key = st.sidebar.text_input("Groq API Key", type="password")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, CSV, DOCX)", type=["pdf", "csv", "docx"])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to DecisionLens! How can I assist you with your business strategy today?"}]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    def process_document(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.type == "text/csv":
                loader = CSVLoader(tmp_file_path)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = Docx2txtLoader(tmp_file_path)
            else:
                st.error("Unsupported file type")
                return None

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings()
            vector_store = FAISS.from_documents(texts, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return None
        finally:
            os.unlink(tmp_file_path)

    # Process uploaded document
    if uploaded_file:
        with st.spinner("Processing uploaded document..."):
            st.session_state.vector_store = process_document(uploaded_file)
        if st.session_state.vector_store:
            st.success("Document processed and added to the knowledge base!")
        else:
            st.error("Failed to process the document. Please try again.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_query = st.chat_input("Ask about your business strategy or data...")

    def get_table_schema(engine):
        inspector = inspect(engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = [column['name'] for column in inspector.get_columns(table_name)]
            schema[table_name] = columns
        return schema

    def create_schema_visualization(schema):
        nodes = []
        links = []
        for i, (table, columns) in enumerate(schema.items()):
            table_node = {"label": table, "shape": "box", "color": "lightblue"}
            nodes.append(table_node)
            for column in columns:
                column_node = {"label": column, "shape": "ellipse", "color": "lightgreen"}
                nodes.append(column_node)
                links.append((table_node["label"], column_node["label"]))
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = [node["label"] for node in nodes],
            color = [node["color"] for node in nodes],
            ),
            link = dict(
            source = [nodes.index({"label": link[0], "shape": "box", "color": "lightblue"}) for link in links],
            target = [nodes.index({"label": link[1], "shape": "ellipse", "color": "lightgreen"}) for link in links],
            value = [1] * len(links)
            )
        )])
        fig.update_layout(title_text="Database Schema", font_size=10)
        return fig

    if user_query and all(db_config.values()) and api_key:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        try:
            # Set up database connection
            encoded_password = quote(db_config["password"])
            connection_string = f"mysql+mysqlconnector://{db_config['user']}:{encoded_password}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            engine = create_engine(connection_string)
            db = SQLDatabase(engine)

            # Initialize LLM and agents
            llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", streaming=True)
            
            # SQL Agent
            sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            sql_agent = create_sql_agent(
                llm=llm,
                toolkit=sql_toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )

            # Vector DB Agent (if document is uploaded)
            if st.session_state.vector_store:
                vector_qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )

            # Generate combined response
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                
                # Get SQL insights
                sql_response = sql_agent.run(user_query, callbacks=[st_callback])
                
                # Get Vector DB insights (if available)
                vector_response = ""
                if st.session_state.vector_store:
                    vector_response = vector_qa.run(user_query)
                
                # Combine insights
                combined_response = f"SQL Database Insights:\n{sql_response}\n\n"
                if vector_response:
                    combined_response += f"Document Analysis Insights:\n{vector_response}\n\n"
                
                combined_response += "Based on these insights, here's a summary for your business strategy:\n"
                final_response = llm.predict(combined_response + "Provide a concise summary of the key points and strategic recommendations based on both the SQL and document analysis insights.")
                
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.write(final_response)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Database Schema")
                schema = get_table_schema(engine)
                schema_fig = create_schema_visualization(schema)
                st.plotly_chart(schema_fig, use_container_width=True)

        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        if not all(db_config.values()) or not api_key:
            st.warning("Please provide all database connection details and Groq API key in the sidebar.")

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I assist you?"}]
        st.session_state.vector_store = None
        st.experimental_rerun()


def main():
    if 'user' not in st.session_state:
        login_signup()
    else:
        landing_page()

if __name__ == "__main__":
    main()
