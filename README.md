# 💰 AI Financial Assistant App

An intelligent **AI-powered financial assistant** built using **Streamlit, LangGraph, and LLMs** to help users manage their finances, track expenses, analyze budgets, and plan financial goals.

## 🚀 Features

- 👤 User Profile Management (income, goals, risk tolerance)
- 📊 Expense Tracking & Categorization
- 📉 Budget Summary with insights
- 📈 Stock Price Lookup (Alpha Vantage API)
- 🏡 Savings Goal Calculator (e.g., house down payment)
- 💳 Loan Repayment Calculator
- 🤖 AI Financial Advice (LLM-powered)
- 🧠 Memory handling (short-term & long-term)

## 🧠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **AI/LLM:** Google Gemini (via LangChain)
- **Workflow Engine:** LangGraph
- **APIs:** Alpha Vantage (Stock Data)
- **Environment Management:** python-dotenv
  
## 📂 Project Structure
FinanceAdvisorApp/
│── FinanceAdvisorApp.py # Main Streamlit App
│── requirements.txt # Dependencies
│── .env # API Keys (not uploaded)
│── README.md # Documentation

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add Environment Variables

Create a .env file and add:

GOOGLE_API_KEY=your_google_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

5️⃣ Run the app
streamlit run FinanceAdvisorApp.py

**💡 Example Queries**

"My name is Alex, I'm 30, income $75000"

"Log: rent $1500, groceries $400"

"How long to save for a $300k house if I save $500/month?"

"What is AAPL stock price?"

"Give me financial advice"

**🧩 How It Works**

Uses intent detection to understand user queries

Routes requests via LangGraph workflow nodes

Maintains:

Short-term memory (session)

Long-term memory (advice history)

Calls APIs + LLM for intelligent responses

**🔒 Security Note**

Do NOT upload .env file

Keep API keys private
