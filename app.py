import base64
import json
from io import BytesIO
import PyPDF2
import re
import random
import string
import streamlit as st
import pdfplumber  # A better PDF text extractor
import google.generativeai as genai  # Import Gemini API
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
import calendar

# Setting Streamlit page configuration
st.set_page_config(page_title="FinExtract", page_icon="📖", layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stToolbar"] {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add main container with gradient background
st.markdown('<div class="main">', unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Function to convert PDF to structured text using pdfplumber
def convert_pdf_to_structured_text(pdf_file):
    """Extracts text from a PDF file and returns it with basic formatting."""
    if not pdf_file:
        st.error("No file uploaded. Please upload a valid PDF file.")  # Show error if no file
        return ""  # Return empty string if no file uploaded

    try:
        with BytesIO(pdf_file.read()) as buffer:  # Use BytesIO to read file-like object
            with pdfplumber.open(buffer) as pdf:
                num_pages = len(pdf.pages)

                if num_pages == 0:
                    st.error("Uploaded file is empty. Please upload a valid PDF file.")  # Error for empty PDF
                    return ""  # Return empty if PDF is empty

                text = ""
                for page_num in range(num_pages):
                    page = pdf.pages[page_num]
                    text += page.extract_text() + "\n\n"  # Add double line breaks for page breaks

                return text
    except Exception as e:
        st.error(f"Error reading the PDF file: {e}")  # Error message for any other issues
        return ""  # Return empty string in case of error

# Setting up Gemini API key
gemini_api_key = st.secrets["API_KEY"]  # Replace with your Gemini API key

# Initialize Gemini client
genai.configure(api_key=gemini_api_key)

# Generate a unique key based on the uploaded file's name or a random string
def generate_unique_key(uploaded_file):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))  # Generate random string
    return f"pdf_viewer_{uploaded_file.name}_{random_str}"

# Function to parse bank statement
def parse_bank_statement_with_gemini(text):
    """Parses the bank statement using Gemini to extract structured data."""
    if not text:
        return None

    prompt = f"""
    You are an expert in analyzing bank statements.
    Your task is to extract key information from the provided text of a bank statement.
    The response should be a valid JSON object. Do not return any code or any comments outside the JSON.
    The json object should have following information as keys:
        -   "bank_name": The name of the bank.
        -   "customer_name": The full name of the account holder.
        -   "account_number": The account number.
        -   "statement_start_date": The starting date of the statement.
        -   "statement_end_date": The end date of the statement.
        -   "starting_balance": The starting balance of the statement.
        -   "total_money_in": Total amount received in the statement period.
        -   "total_money_out": The total amount spent during the statement period.
        -   "ending_balance": The closing balance in the statement.
        -   "transactions" : an array with elements as object. Each object containing transaction info. The info is date, description, money_out, money_in and balance

    If a piece of information can not be retrieved please fill "N/A" as the value of that key.
    Here is the bank statement text:
    ```
    {text}
    ```
    """
    
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        if response and response.text.strip():
            # Attempt to parse the JSON response
            try:
                 cleaned_response = response.text.replace('```json','').replace('```','') # Remove any code formatting
                 return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse the response: {e}, Raw: {response.text}")
                return None

        else:
            st.error("The API response is empty or invalid.")
            return None
    except Exception as e:
         st.error(f"Failed to get response from Gemini API: {e}")
         return None
        
def process_data(data):
    """Processes the data from Gemini for display and visualization."""
    if not data:
        return None, None, None, None

    summary_data = {
      "bank_name": data.get("bank_name", "N/A"),
      "customer_name": data.get("customer_name", "N/A"),
      "account_number": data.get("account_number", "N/A"),
      "statement_start_date": data.get("statement_start_date", "N/A"),
      "statement_end_date": data.get("statement_end_date", "N/A"),
      "starting_balance": data.get("starting_balance", "N/A"),
      "total_money_in": data.get("total_money_in", "N/A"),
      "total_money_out": data.get("total_money_out", "N/A"),
      "ending_balance": data.get("ending_balance", "N/A")
    }
    
    transactions = data.get("transactions", [])
    df = None
    if transactions:
      df = pd.DataFrame(transactions)
      df['date'] = pd.to_datetime(df['date'], errors='coerce')
      df['money_out'] = pd.to_numeric(df['money_out'], errors='coerce').fillna(0)
      df['money_in'] = pd.to_numeric(df['money_in'], errors='coerce').fillna(0)
    
    return summary_data, df, transactions

def generate_financial_summary(summary_data, df, time_frame):
    """Generates a detailed financial summary using Gemini."""

    if not summary_data or df is None or df.empty:
        return "No data available for financial summary."

    if time_frame:
          summary_text = f"""
            Here is the financial data:
            Bank Name: {summary_data.get('bank_name', 'N/A')}
            Customer Name: {summary_data.get('customer_name', 'N/A')}
            Account Number: {summary_data.get('account_number', 'N/A')}
            Statement Start Date: {summary_data.get('statement_start_date', 'N/A')}
            Statement End Date: {summary_data.get('statement_end_date', 'N/A')}
            Starting Balance: £{summary_data.get('starting_balance', 'N/A').replace(",", "")}
            Total Money In: £{summary_data.get('total_money_in', 'N/A').replace(",", "")}
            Total Money Out: £{summary_data.get('total_money_out', 'N/A').replace(",", "")}
            Ending Balance: £{summary_data.get('ending_balance', 'N/A').replace(",", "")}

            Transactions:
            {df.to_markdown(index = False)}

            Analyze the provided financial data and transactions for {time_frame}, then generate a detailed financial summary. The summary should include:
                - An overview of financial health.
                - Income analysis with sources.
                - Spending analysis with major categories.
                - Highest spending and income categories
                - Notable recurring items like income and expenses
            Provide key insights and recommendations in points. Make the output very well structured and formatted using markdown
        """

    else:
        summary_text = f"""
            Here is the financial data:
            Bank Name: {summary_data.get('bank_name', 'N/A')}
            Customer Name: {summary_data.get('customer_name', 'N/A')}
            Account Number: {summary_data.get('account_number', 'N/A')}
            Statement Start Date: {summary_data.get('statement_start_date', 'N/A')}
            Statement End Date: {summary_data.get('statement_end_date', 'N/A')}
            Starting Balance: £{summary_data.get('starting_balance', 'N/A').replace(",", "")}
            Total Money In: £{summary_data.get('total_money_in', 'N/A').replace(",", "")}
            Total Money Out: £{summary_data.get('total_money_out', 'N/A').replace(",", "")}
            Ending Balance: £{summary_data.get('ending_balance', 'N/A').replace(",", "")}

            Transactions:
            {df.to_markdown(index = False)}

            Analyze the provided financial data and transactions, then generate a detailed financial summary. The summary should include:
                - An overview of financial health.
                - Income analysis with sources.
                - Spending analysis with major categories.
                - Highest spending and income categories
                - Notable recurring items like income and expenses
            Provide key insights and recommendations in points. Make the output very well structured and formatted using markdown
        """


    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(summary_text)
        if response and response.text.strip():
            return response.text
        else:
            return "Could not generate a financial summary."
    except Exception as e:
        return f"Failed to get response from Gemini API: {e}"

def create_pdf_report(summary, filename, time_frame = None):
    """Creates a PDF report of the financial summary."""

    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

      # Custom title style
    title_style = ParagraphStyle(
      'Title',
      parent = styles['h1'],
      fontSize=20,
      alignment=TA_LEFT,
      textColor=colors.navy
      )

    if time_frame:
        story.append(Paragraph(f"Financial Summary for {time_frame}", title_style))
    else:
       story.append(Paragraph("Financial Summary", title_style))


    # Custom text style
    text_style = ParagraphStyle(
        'Text',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        alignment=TA_JUSTIFY
    )
    
    story.append(Paragraph(summary, text_style))

    doc.build(story)


# Create tabs at the top
tab1, tab2, tab3 = st.tabs(["Home", "AI Service", "Contact Us"])

# Add title and subtitle
st.markdown('<h1 class="title">FinExtract</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">This app helps you extract information from your bank statement PDF.</p>', unsafe_allow_html=True)

# Handle tab content
with tab1:
    st.write("Welcome to FinExtract!")
    st.write("""This application helps you extract key information from your bank statements quickly and efficiently.
    Navigate to the AI Service tab to upload your bank statement and get started.""")

with tab2:
    uploaded_file = st.file_uploader(
        "Upload your bank statement PDF",
        type=["pdf"],
        key="pdf_uploader"
    )

    if uploaded_file:
        structured_text = convert_pdf_to_structured_text(uploaded_file)
        
        if structured_text:
            with st.spinner("Analyzing Bank Statement ..."):
              parsed_data = parse_bank_statement_with_gemini(structured_text)

            summary_data, df, transactions = process_data(parsed_data)
            time_frame = None
            if summary_data and df is not None and not df.empty:
                if df['date'].dt.year.nunique() > 1:
                  unique_years = sorted(df['date'].dt.year.unique(), reverse = True)
                  time_frame = st.select_slider("Select time frame", options = ["All time"] + [f"{year}" for year in unique_years] + [f"{calendar.month_name[month]}-{year}" for year in unique_years for month in range(1,13)])
                  if time_frame != "All time":
                    if time_frame.find("-") != -1:
                      month, year = time_frame.split("-")
                      df = df[(df['date'].dt.year == int(year)) & (df['date'].dt.month == list(calendar.month_name).index(month))]
                      time_frame = f"{calendar.month_name[list(calendar.month_name).index(month)]} of {year}"
                    else:
                      df = df[df['date'].dt.year == int(time_frame)]
                      time_frame = f"Year {time_frame}"

            financial_summary = None
            if summary_data and df is not None and not df.empty:
                with st.spinner("Generating Financial Summary"):
                   financial_summary = generate_financial_summary(summary_data, df, time_frame)

            # Display PDF and results in two columns
            col1, col2 = st.columns(spec=[2, 1], gap="small")

            with col1:
                with st.container(border=True):
                    # Optionally render the PDF viewer
                     st.text("PDF viewer would go here.")

                if financial_summary:
                    with st.expander("Financial Summary", expanded=True):
                        st.write(financial_summary)

                        # Download PDF report
                        report_filename = "financial_report.pdf"
                        create_pdf_report(financial_summary, report_filename, time_frame)
                        with open(report_filename, "rb") as f:
                            pdf_bytes = f.read()

                        st.download_button(
                            label="Download Financial Report (PDF)",
                            data=pdf_bytes,
                            file_name=report_filename,
                            mime="application/pdf",
                        )

                if df is not None and not df.empty:
                    with st.expander("Monthly Spending Chart"):
                        monthly_spending = df.groupby(df['date'].dt.month).agg({'money_out':'sum', 'money_in':'sum'})
                        monthly_spending.index = [calendar.month_name[month] for month in monthly_spending.index]
                        monthly_spending = monthly_spending.reindex([calendar.month_name[month] for month in range(1,13)], fill_value=0)
                        fig = px.line(monthly_spending, x = monthly_spending.index, y = ['money_out', 'money_in'])
                        st.plotly_chart(fig)


                    with st.expander("Category Spending Chart"):
                        category_spending = df.groupby('description')['money_out'].sum()
                        fig = px.pie(names=category_spending.index, values=category_spending.values, hole = 0.3)
                        fig.update_traces(textposition='outside', textinfo='percent+label')
                        st.plotly_chart(fig)

            with col2:
                if summary_data:
                    with st.expander("Customer Information", expanded=True):
                        # Display extracted information
                        labels = ["Bank Name", "Customer Name", "Account Number", "Statement Start Date", "Statement End Date", "Starting Balance", "Total Money In", "Total Money Out", "Ending Balance"]
                        for label, key in zip(labels, summary_data):
                            st.markdown(f"<p style='font-family: Arial; color: black; font-size: 20px;'>{label}:</p>", unsafe_allow_html=True)
                            st.write(summary_data[key] if summary_data[key] != "N/A" else "N/A")

                    if df is not None and not df.empty:
                        with st.expander("Transactions Table", expanded=True):
                            st.dataframe(df)
                else:
                    st.error("No data found.")
            
            # Add a chat interface at bottom
            if structured_text:
               st.markdown("<hr>", unsafe_allow_html=True)
               st.subheader("Chat with your bank statement")
               query = st.text_input("Enter your question:")
               if query:
                   chat_prompt = f"""
                       You are a helpful assistant, skilled at understanding and responding to questions regarding bank statements.
                       Here is a bank statement text : "{structured_text}"
                       Here is the user query: "{query}"
                       Provide a response in markdown format.
                     """
                   try:
                        chat_response = genai.GenerativeModel("gemini-1.5-flash").generate_content(chat_prompt)
                        if chat_response and chat_response.text.strip():
                            st.write(chat_response.text)
                        else:
                            st.write("Could not generate a response.")
                   except Exception as e:
                       st.error(f"Error getting response from Gemini API: {e}")

with tab3:
    st.write("### Contact Information")
    st.write("For support or inquiries, please reach out to us:")
    st.write("📧 Email: support@finextract.com")
    st.write("📞 Phone: +1-XXX-XXX-XXXX")
    st.write("🌐 Website: www.finextract.com")