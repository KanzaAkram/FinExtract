# Bank Statement Data Extraction App

  This repository contains a Streamlit application powered by GPT, designed to extract data from bank statements in PDF format. The app utilizes the ChatGPT API to read and comprehend the bank statement, providing essential information such as the highest credit and debit amounts, total balance, customer name, account number, and more.

  Check the app at : https://bankstatement-data-extractor-app.streamlit.app/
  
<div style="text-align: center;">
    <img src="https://github.com/GhufranBarcha/BankStatement-Data-Extractor/blob/main/images/image1.png" alt="Alt text" width="700" height="400">
</div>
<div style="text-align: center;">
    <img src="https://github.com/GhufranBarcha/BankStatement-Data-Extractor/blob/main/images/image2.png" alt="Alt text" width="700" height="400">
</div>

## Key Features:

- Upload and process bank statement PDF files.
- Utilize the ChatGPT API to analyze and interpret the bank statement content.
- Extract crucial data points including highest credit and debit amounts, total balance, customer name, and account number.
- User-friendly interface for easy interaction and data retrieval.
## How to Use:

1. Upload your bank statement PDF file using the provided interface.
2. The app will process the PDF content using the ChatGPT API to extract relevant data.
3. Once processing is complete, the extracted information will be displayed, including highest credit and debit amounts, total balance, customer name, and account number.
4. Users can easily access and utilize the extracted data for further analysis or record-keeping purposes.
## Technology Stack:

- Streamlit: A Python library for creating interactive web applications.
- ChatGPT API: OpenAI's language model used for understanding and analyzing text-based content.
- PyPDF2: Python library for reading PDF files and extracting text content.
- JSON: Data interchange format used for communication between the app and the ChatGPT API.
## Usage:

- Clone the repository to your local machine.
- Install the required dependencies listed in the requirements.txt file.
- Run the Streamlit application using the provided command.
- Upload your bank statement PDF file and interact with the app to extract and analyze the data.


Step 1: Create a Virtual Environment
Create a virtual environment in your Codespaces workspace:

bash
Copy code
python3 -m venv venv
Step 2: Activate the Virtual Environment
Activate the environment:

bash
Copy code
source venv/bin/activate
If the above command works, your shell prompt should show something like (venv).

Step 3: Install Dependencies
Now, install the required dependencies:

If you have requirements.txt:
bash
Copy code
pip install -r requirements.txt
If streamlit is not listed in requirements.txt: Install it manually:
bash
Copy code
pip install streamlit
Step 4: Run the Streamlit App
Run the app using:

bash
Copy code
streamlit run app.py