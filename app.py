import streamlit as st
import pandas as pd
import os

# LangChain imports - minimal set
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType

# Set page configuration
st.set_page_config(
    page_title="CSV Analyzer", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stButton button {
        width: 100%;
    }
    .st-bd {
        padding-top: 3rem;
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 10rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-bottom: 3.5rem;
        padding-left: 1rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Function to get OpenAI API key
def get_openai_api_key():
    """Get OpenAI API key from environment or user input"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            st.sidebar.success("API Key provided!")
    return api_key

# Function to set up the agent
def setup_agent(df, api_key):
    """Set up a pandas agent to analyze the dataframe"""
    try:
        # Create the language model
        llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        
        # Define the agent prompt
        agent_prompt = """You are a data analyst working with a pandas DataFrame.
        The DataFrame is available as the variable `df`.
        
        Analyze this DataFrame to answer the user's questions.
        
        Guidelines:
        1. For questions about the dataset structure:
           - Use df.shape, df.columns, df.dtypes, df.info()
           - Summarize the schema concisely
        
        2. For statistical questions:
           - Use df.describe(), df.mean(), df.median(), etc.
           - For categorical columns, use df[column].value_counts()
           - Be precise with statistics (exact figures)
        
        3. For more complex analysis:
           - Write clear, efficient pandas code
           - Use groupby, filters, and aggregations as needed
           - Explain your findings in plain language after the code
        
        Always include your code in Python code blocks (```python).
        DO NOT invent data or features that don't exist in the dataframe.
        """
        
        # Create the pandas agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=agent_prompt
        )
        
        return agent
    
    except Exception as e:
        st.error(f"Error setting up agent: {str(e)}")
        return None

# Function to handle file upload
@st.cache_data
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Initialize session state variables
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.title("CSV Analyzer")
    st.sidebar.markdown("---")
    
    # Get OpenAI API key
    api_key = get_openai_api_key()
    
    # File uploader section
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # Sample data section
    with st.sidebar.expander("Don't have a CSV?"):
        if st.button("Generate Sample Data"):
            import numpy as np
            # Create sample data
            sample_data = pd.DataFrame({
                'ID': range(1, 101),
                'Age': np.random.randint(18, 70, 100),
                'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], 100),
                'Income': np.random.normal(50000, 15000, 100).round(2),
                'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
                'Years_Experience': np.random.randint(0, 30, 100),
                'Department': np.random.choice(['Sales', 'Marketing', 'HR', 'Engineering', 'Support'], 100),
                'Performance_Score': np.random.normal(7, 1.5, 100).round(2)
            })
            
            # Add correlated columns
            sample_data['Salary'] = (
                sample_data['Years_Experience'] * 2000 + 
                sample_data['Performance_Score'] * 5000 + 
                np.random.normal(30000, 5000, 100)
            ).round(2)
            
            # Save as CSV
            csv = sample_data.to_csv(index=False)
            
            # Provide download button
            st.sidebar.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="sample_employee_data.csv",
                mime="text/csv"
            )
    
    # Clear chat button
    if st.session_state.messages:
        if st.sidebar.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Main area
    if uploaded_file is not None:
        # Load the data
        df = load_csv(uploaded_file)
        
        if df is not None:
            # Store in session state
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            
            # Header area
            st.title(f"📊 Analyzing: {uploaded_file.name}")
            st.markdown("Ask questions about your data in natural language")
            
            # Data preview
            with st.expander("Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column information
                st.subheader("Column Information")
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Display column types
                col_info = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null Values": df.count().values,
                    "Null Values": df.isna().sum().values,
                    "Unique Values": [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Chat interface
            if api_key:
                # Set up agent
                agent = setup_agent(df, api_key)
                
                if agent:
                    # Example questions
                    with st.expander("Example Questions"):
                        st.markdown("""
                        Try asking questions like:
                        - What's the shape and structure of this dataset?
                        - Summarize the basic statistics for all columns
                        - What's the average, median, and standard deviation of [column]?
                        - How many unique values are in [column]?
                        - Find the correlation between [column1] and [column2]
                        - Group by [categorical column] and calculate average [numeric column]
                        - What rows have [column] greater than [value]?
                        - Which department has the highest average salary?
                        """)
                    
                    # Display chat history
                    st.subheader("Chat with your data")
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    
                    # Chat input
                    if prompt := st.chat_input("Ask a question about your data..."):
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        # Get response from agent
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing data..."):
                                try:
                                    response = agent.run(prompt)
                                    st.markdown(response)
                                    
                                    # Add assistant response to chat history
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": response
                                    })
                                except Exception as e:
                                    error_msg = f"Error analyzing data: {str(e)}"
                                    st.error(error_msg)
                                    
                                    # Add error to chat history
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": error_msg
                                    })
                else:
                    st.error("Failed to initialize the analysis agent. Please check your API key.")
            else:
                st.warning("Please provide an OpenAI API key to enable the chat functionality.")
    else:
        # Welcome screen
        st.title("📊 CSV Analyzer")
        st.markdown("""
        ## Chat with your CSV data using natural language
        
        Upload a CSV file to get started!
        
        This app lets you:
        - Analyze any CSV file with natural language questions
        - Get instant statistics and insights
        - Perform complex data analysis without writing code
        - Explore relationships and patterns in your data
        
        Simply upload your file using the sidebar on the left.
        """)
        
        # Feature showcases with columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📈 Basic Analysis")
            st.markdown("""
            - Dataset structure
            - Column statistics
            - Missing values
            - Data types
            """)
            
        with col2:
            st.markdown("### 🔍 Advanced Analysis")
            st.markdown("""
            - Correlations
            - Group by operations
            - Filtering data
            - Finding outliers
            """)
            
        with col3:
            st.markdown("### 💡 Natural Language")
            st.markdown("""
            - Ask in plain English
            - No coding required
            - Instant insights
            - Export results
            """)

if __name__ == "__main__":
    main()
