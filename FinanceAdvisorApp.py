import os
import json
import math
import asyncio
import streamlit as st
import re
import requests
import logging
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # your Google LLM wrapper
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional
from typing_extensions import TypedDict  # for state schema

# Load environment variables
load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("financial_assistant")

# --- API keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # optional, used for stock info

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")

# --- Small helper utilities ---
def repair_json(possible_json: str) -> str:
    """
    Basic repair: convert single quotes to double quotes, remove trailing text after last brace,
    remove trailing commas before closing brace, and ensure booleans/null are lower-case JSON.
    This is a pragmatic helper — not a full JSON fixer.
    """
    if not possible_json:
        return possible_json
    # trim to last closing brace
    last_brace = possible_json.rfind("}")
    if last_brace != -1:
        possible_json = possible_json[: last_brace + 1]
    # replace single quotes with double quotes (naive)
    s = possible_json.strip()
    # avoid converting quotes inside already-correct JSON (best-effort)
    s = s.replace("'", '"')
    # remove trailing commas before } and ]
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    # convert Python-style True/False/None to JSON equivalents
    s = re.sub(r"\bNone\b", "null", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x.replace('$', '').replace(',', '').strip())
        return default
    except Exception:
        return default

# --- Configure the Google LLM wrapper ---
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    # Some wrappers may not accept None for these fields; adjust as necessary for your environment.
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # If the ChatGoogleGenerativeAI requires the API key explicitly, pass it here (depends on your wrapper).
    # api_key=GOOGLE_API_KEY
)

# === State Definition (TypedDict schema used by StateGraph) ===
class FinanceStateSchema(TypedDict, total=False):
    user_input: str
    intent: str
    data: Dict[str, Any]
    user_profile: Dict[str, Any]
    short_term_memory: Dict[str, Any]
    long_term_memory: Dict[str, Any]

# Runtime state typing alias (just for readability in code)
FinanceState = Dict[str, Any]

# --- 2. NODE (handler) FUNCTIONS ---
async def collect_user_data(state: FinanceState) -> FinanceState:
    user_input = state.get('user_input', '')
    user_profile = dict(state.get('user_profile', {}))
    short_term_memory = dict(state.get('short_term_memory', {}))

    extraction_prompt = (
        f"Extract user profile information from this text: {user_input}\n"
        f"Required fields: 'name', 'age', 'income', 'goals', 'risk_tolerance'.\n"
        f"Return only a valid JSON object with these keys. Use null for missing values.\n"
        f"Important: use double quotes for keys and strings.\n"
    )
    try:
        resp = await llm.ainvoke(extraction_prompt)
        text = getattr(resp, "content", str(resp)).strip()
        js_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not js_match:
            raise ValueError("No JSON object detected in LLM output.")
        cleaned = repair_json(js_match.group(0))
        new_profile = json.loads(cleaned)
        if isinstance(new_profile, dict):
            for k, v in new_profile.items():
                if v is not None and v != '' and str(v).lower() != 'none':
                    user_profile[k] = v
            if not user_profile.get('income'):
                message = "I have your profile information, but I need your yearly or monthly income. Can you provide that?"
                short_term_memory['last_question'] = message
            else:
                message = f"Thanks — I've updated your profile: {json.dumps(user_profile, indent=2)}. How can I help next?"
        else:
            raise ValueError("Extracted JSON is not an object/dict.")
    except Exception as e:
        logger.error(f"collect_user_data error: {e}")
        message = (
            "❌ I couldn't understand your profile. Please list your information clearly, for example:\n"
            "'My name is Alex, I'm 30, and my yearly income is $75,000. My goal is to buy a house.'"
        )
    state.update({
        "user_profile": user_profile,
        "short_term_memory": short_term_memory,
        "data": {"response": message}
    })
    return state

async def get_stock_info(state: FinanceState) -> FinanceState:
    user_input = state.get('user_input', '')
    short_term_memory = dict(state.get('short_term_memory', {}))
    user_profile = dict(state.get('user_profile', {}))

    prompt = (
        f"Extract and return only the stock ticker symbol (e.g., 'AAPL') from this text: {user_input}\n"
        f"If you cannot determine a ticker, return UNKNOWN."
    )
    try:
        resp = await llm.ainvoke(prompt)
        text = getattr(resp, "content", str(resp)).strip().upper()
        js_match = re.search(r'([A-Z]{1,5})', text)
        stock_symbol = js_match.group(1) if js_match else ("UNKNOWN" if "UNKNOWN" in text else text)
    except Exception as e:
        logger.error(f"LLM error extracting stock symbol: {e}")
        stock_symbol = "UNKNOWN"

    if stock_symbol == "UNKNOWN" or not re.match(r'^[A-Z]{1,5}$', stock_symbol):
        message = f"Sorry, I couldn't identify a valid stock symbol from '{user_input}'. Please specify the ticker (e.g., 'AAPL')."
        logger.warning(f"Invalid stock symbol extracted: {stock_symbol}")
    else:
        if not ALPHA_VANTAGE_API_KEY:
            message = (
                "Alpha Vantage API key not configured; cannot fetch real stock data. "
                "Set ALPHA_VANTAGE_API_KEY in your .env if you want live quotes."
            )
            logger.warning("No ALPHA_VANTAGE_API_KEY set; skipping stock API call.")
        else:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
                if "Time Series (Daily)" in data:
                    latest_date = sorted(data["Time Series (Daily)"].keys(), reverse=True)[0]
                    stock_data = data["Time Series (Daily)"][latest_date]
                    close_price = stock_data.get("4. close")
                    message = f"The latest closing price for {stock_symbol} is ${close_price} (as of {latest_date})."
                    risk_tolerance = user_profile.get('risk_tolerance', 'unknown')
                    risk_prompt = (
                        f"Give a short, empathetic note about investing in {stock_symbol} for a user with "
                        f"{risk_tolerance} risk tolerance. Keep it brief."
                    )
                    rr = await llm.ainvoke(risk_prompt)
                    message += "\n" + getattr(rr, "content", str(rr)).strip()
                elif "Error Message" in data:
                    message = f"Alpha Vantage error: {data.get('Error Message')}"
                    logger.error(message)
                elif "Note" in data and "rate limit" in data.get("Note", "").lower():
                    message = "Alpha Vantage API rate limit exceeded. Please try again shortly."
                    logger.warning("Alpha Vantage rate limit hit.")
                else:
                    message = f"No daily time series data available for {stock_symbol}."
                    logger.error(f"Unexpected Alpha Vantage response: {data}")
            except requests.RequestException as e:
                message = f"Error fetching stock data for {stock_symbol}: {e}"
                logger.error(message)

    short_term_memory['last_stock_requested'] = user_input
    state.update({"short_term_memory": short_term_memory, "data": {"response": message}})
    return state

async def process_expenses(state: FinanceState) -> FinanceState:
    user_input = state.get("user_input", "")
    short_term_memory = dict(state.get("short_term_memory", {}))

    # Pull existing expenses if present
    current_expenses = dict(short_term_memory.get("expenses", {}))

    extraction_prompt = (
        f"Extract a JSON object of monthly expenses from this text: {user_input}\n"
        f"Fields: 'rent','groceries','transportation','utilities','entertainment','subscriptions','savings','other'.\n"
        f"Return a JSON object with numeric values or null for missing ones."
    )
    try:
        resp = await llm.ainvoke(extraction_prompt)
        text = getattr(resp, "content", str(resp)).strip()
        js_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not js_match:
            raise ValueError("No JSON found in LLM output")
        new_expenses = json.loads(repair_json(js_match.group(0)))
        if not isinstance(new_expenses, dict):
            raise ValueError("Parsed expenses are not a dict")
        # Merge: overwrite or add
        for k, v in new_expenses.items():
            if v is not None:
                # coerce to float if possible
                try:
                    current_expenses[k] = float(v)
                except Exception:
                    current_expenses[k] = v
        short_term_memory["expenses"] = current_expenses
        message = "✅ I have successfully logged your expenses. I will use this data for future budget summaries."
    except Exception as e:
        logger.error(f"process_expenses error: {e}")
        message = (
            "❌ I couldn't understand your expenses. Please list them clearly, e.g.: "
            "'My rent is $1500, groceries $450, transportation $200, utilities $250, entertainment $150, subscriptions $50, savings $500.'"
        )

    state.update({"short_term_memory": short_term_memory, "data": {"response": message}})
    return state

async def budget_summary(state: FinanceState) -> FinanceState:
    user_profile = dict(state.get('user_profile', {}))
    user_expenses = dict(state.get('short_term_memory', {}).get('expenses', {}))

    if not user_expenses:
        message = ("I don't see any expense data. Could you provide monthly expenses, e.g. "
                   "'My rent is $1500, groceries $450, utilities $200'?")
        state["data"] = {"response": message}
        return state

    income_value = user_profile.get("income")
    if not income_value:
        message = "I have your expenses, but I need your monthly or yearly income to provide a complete summary. Please provide it."
        state["data"] = {"response": message}
        return state

    try:
        annual_income = safe_float(income_value)
        monthly_income = annual_income / 12.0
    except Exception as e:
        logger.error(f"Error parsing income: {e}")
        state["data"] = {"response": "❌ I couldn't parse your income. Please supply a numeric yearly income (e.g., 75000)."}
        return state

    total_expenses = sum(v for v in user_expenses.values() if isinstance(v, (int, float, float)))
    breakdown_lines = []
    for k, v in user_expenses.items():
        pct = (v / total_expenses * 100) if total_expenses > 0 else 0
        breakdown_lines.append(f"- {k}: ${v:,.2f} ({pct:.1f}% of expenses)")

    prompt = (
        f"Provide a concise budget summary given monthly income ${monthly_income:,.2f}, "
        f"monthly expenses {user_expenses}, total expenses ${total_expenses:,.2f}. "
        f"Keep advice short and empathetic."
    )
    try:
        response = await llm.ainvoke(prompt)
        llm_text = getattr(response, "content", str(response)).strip()
        message = f"Monthly income: ${monthly_income:,.2f}\nTotal expenses: ${total_expenses:,.2f}\n\n" \
                  + "\n".join(breakdown_lines) + "\n\n" + llm_text
    except Exception as e:
        logger.error(f"budget_summary LLM error: {e}")
        message = "Sorry, I couldn't generate a budget summary at this time."

    state["data"] = {"response": message}
    return state

async def provide_advice(state: FinanceState) -> FinanceState:
    user_input = state.get('user_input', '')
    user_profile = dict(state.get('user_profile', {}))
    long_term_memory = dict(state.get('long_term_memory', {}))

    prompt = (
        f"Provide simple, empathetic financial advice for this request: {user_input}\n"
        f"User profile: {user_profile}\n"
        f"Previous advice: {long_term_memory.get('last_advice', 'none')}\n"
        f"Keep it accessible for users with limited financial literacy."
    )
    try:
        resp = await llm.ainvoke(prompt)
        message = getattr(resp, "content", str(resp)).strip()
        long_term_memory['last_advice'] = message
    except Exception as e:
        logger.error(f"provide_advice error: {e}")
        message = "Sorry, I couldn't generate advice at this moment."

    state.update({"long_term_memory": long_term_memory, "data": {"response": message}})
    return state

async def calculate_savings_plan(state: FinanceState) -> FinanceState:
    user_input = state.get("user_input", "")
    short_term_memory = dict(state.get("short_term_memory", {}))

    extraction_prompt = (
        f"From this text: {user_input}, extract a JSON with keys: savings_per_month, home_cost, goal_years, interest_rate. "
        f"Return only JSON."
    )
    try:
        resp = await llm.ainvoke(extraction_prompt)
        text = getattr(resp, "content", str(resp)).strip()
        js_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not js_match:
            raise ValueError("No JSON in LLM output")
        values = json.loads(repair_json(js_match.group(0)))
        # fallback to short_term_memory values
        savings = safe_float(values.get("savings_per_month", short_term_memory.get("savings_per_month", 0)))
        home_cost = safe_float(values.get("home_cost", short_term_memory.get("home_cost", 0)))
        years = safe_float(values.get("goal_years", short_term_memory.get("goal_years", 0)))
        interest_pct = safe_float(values.get("interest_rate", short_term_memory.get("interest_rate", 0)))
        interest = interest_pct / 100.0

        short_term_memory.update({
            "savings_per_month": savings,
            "home_cost": home_cost,
            "goal_years": years,
            "interest_rate": interest_pct
        })

        if savings <= 0 and home_cost > 0 and years > 0:
            # compute required monthly savings to reach 20% down payment in given years
            down_payment = home_cost * 0.20
            if interest > 0:
                monthly_rate = interest / 12
                required_savings = (down_payment * monthly_rate) / ((1 + monthly_rate)**(years * 12) - 1)
            else:
                required_savings = down_payment / (years * 12)
            message = (
                f"To reach a 20% down payment (${down_payment:,.0f}) for a ${home_cost:,.0f} home in {int(years)} years, "
                f"you'd need to save about ${required_savings:,.0f} per month."
            )
            state.update({"short_term_memory": short_term_memory, "data": {"response": message}})
            return state

        if savings <= 0:
            message = "⚠️ Please provide a valid monthly savings amount to calculate your plan."
        elif home_cost <= 0:
            message = "⚠️ Please provide a valid home cost to calculate your plan."
        else:
            down_payment = home_cost * 0.20
            if interest > 0:
                monthly_rate = interest / 12
                if (savings * (1 + monthly_rate)) <= (monthly_rate * down_payment):
                    message = "❌ At your current savings rate and interest, your monthly savings are too low to ever reach your down payment goal."
                else:
                    months_needed = math.log(
                        (savings * (1 + monthly_rate)) /
                        ((savings * (1 + monthly_rate)) - (monthly_rate) * down_payment)
                    ) / math.log(1 + monthly_rate)
                    result_years = round(months_needed / 12, 2)
                    message = (
                        f"To buy a ${home_cost:,.0f} home with a 20% down payment (${down_payment:,.0f}),\n"
                        f"and save ${savings:,.0f} per month{' at ' + str(interest_pct) + '% interest' if interest > 0 else ''},\n"
                        f"you will reach your down payment in approximately {result_years} years."
                    )
            else:
                months_needed = down_payment / savings
                result_years = round(months_needed / 12, 2)
                message = (
                    f"To buy a ${home_cost:,.0f} home with a 20% down payment (${down_payment:,.0f}),\n"
                    f"and save ${savings:,.0f} per month,\n"
                    f"you will reach your down payment in approximately {result_years} years."
                )
            if years > 0:
                if result_years <= years:
                    message += "\n🎉 You’re on track to achieve your goal!"
                else:
                    message += f"\n⚠️ You may need to increase savings or delay your purchase to meet your goal of {int(years)} years."
    except Exception as e:
        logger.error(f"calculate_savings_plan error: {e}")
        message = ("❌ Couldn't analyze savings plan. Please provide the home cost, monthly savings, and goal years. "
                   "Example: 'I want to save for a $300000 house, I can save $500 a month and want to do it in 5 years.'")

    state.update({"short_term_memory": short_term_memory, "data": {"response": message}})
    return state

async def calculate_loan_payment(state: FinanceState) -> FinanceState:
    user_input = state.get("user_input", "")
    short_term_memory = dict(state.get("short_term_memory", {}))

    extraction_prompt = (
        f"From this text: {user_input}, extract JSON with keys loan_amount, monthly_payment, interest_rate. Return only JSON."
    )
    try:
        resp = await llm.ainvoke(extraction_prompt)
        text = getattr(resp, "content", str(resp)).strip()
        js_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not js_match:
            raise ValueError("No JSON in LLM output")
        values = json.loads(repair_json(js_match.group(0)))
        loan_amount = safe_float(values.get("loan_amount", short_term_memory.get("loan_amount", 0)))
        monthly_payment = safe_float(values.get("monthly_payment", short_term_memory.get("monthly_payment", 0)))
        interest_rate_pct = safe_float(values.get("interest_rate", short_term_memory.get("interest_rate", 0)))
        interest_rate = interest_rate_pct / 100.0

        short_term_memory.update({
            "loan_amount": loan_amount,
            "monthly_payment": monthly_payment,
            "interest_rate": interest_rate_pct
        })

        if loan_amount <= 0:
            message = "⚠️ Please provide a valid loan amount."
        elif monthly_payment <= 0:
            message = "⚠️ Please provide a valid monthly payment."
        elif interest_rate <= 0:
            message = "💡 Please provide the interest rate to calculate repayment time."
        else:
            monthly_rate = interest_rate / 12
            if monthly_payment <= loan_amount * monthly_rate:
                message = "❌ Your monthly payment is too low to ever pay off the loan. It only covers the interest."
            else:
                months_needed = math.log(
                    monthly_payment / (monthly_payment - loan_amount * monthly_rate)
                ) / math.log(1 + monthly_rate)
                years_needed = round(months_needed / 12, 2)
                message = (
                    f"💳 Loan amount: ${loan_amount:,.0f}\n"
                    f"💵 Monthly payment: ${monthly_payment:,.0f}\n"
                    f"📈 Interest rate: {interest_rate_pct:.2f}%\n"
                    f"⏳ Time to repay: {months_needed:.0f} months (~{years_needed} years)"
                )
    except Exception as e:
        logger.error(f"calculate_loan_payment error: {e}")
        message = ("❌ Couldn't calculate loan repayment. Please provide the loan amount, monthly payment, and interest rate. "
                   "Example: 'How long will it take to pay off a $20,000 loan with a $400 monthly payment at 6.5% interest?'")

    state.update({"short_term_memory": short_term_memory, "data": {"response": message}})
    return state

async def human_in_the_loop(state: FinanceState) -> FinanceState:
    state["data"] = {"response": "Human intervention required."}
    return state

async def fallback(state: FinanceState) -> FinanceState:
    state["data"] = {"response":
        "Sorry, I didn't understand that. I can help with profile updates, stock lookups, logging expenses, budgets, savings plans, loan calculations, and general advice. "
        "Try one of these examples:\n- 'My name is Alex, I'm 30, yearly income $75000.'\n- 'Log: rent $1500, groceries $450.'\n- 'How long to save for a $350k house if I save $500/month?'\n- 'What's AAPL trading at?'\n"
    }
    return state

# === INTENT DETECTION ===
async def detect_intent(state: FinanceState) -> FinanceState:
    user_input = state.get("user_input", "")
    previous_intent = state.get("short_term_memory", {}).get("previous_intent", "none")

    prompt = (
        f"Classify the user's intent into one of: profile, stock, log_expense, budget, advice, savings_plan, loan_calculation, unknown.\n"
        f"User input: {user_input}\n"
        f"Previous intent: {previous_intent}\n"
        f"Return only one word from the list above."
    )
    try:
        resp = await llm.ainvoke(prompt)
        text = getattr(resp, "content", str(resp)).strip().lower()
        # Extract first word composed of letters/underscores
        m = re.search(r"(profile|stock|log_expense|budget|advice|savings_plan|loan_calculation|unknown)", text)
        intent = m.group(1) if m else "unknown"
    except Exception as e:
        logger.error(f"detect_intent error: {e}")
        intent = "unknown"

    short_term_memory = dict(state.get("short_term_memory", {}))
    short_term_memory["previous_intent"] = intent
    state.update({"intent": intent, "short_term_memory": short_term_memory})
    logger.info(f"Detected intent: {intent}")
    return state

# --- 3. LANGGRAPH ARCHITECTURE ---
# Pass the required state_schema parameter to StateGraph
builder = StateGraph(state_schema=FinanceStateSchema)

builder.add_node("Intent Detection", detect_intent)
builder.add_node("Collect User Data", collect_user_data)
builder.add_node("Stock Info", get_stock_info)
builder.add_node("Process Expenses", process_expenses)
builder.add_node("Budget Summary", budget_summary)
builder.add_node("Provide Advice", provide_advice)
builder.add_node("Savings Plan", calculate_savings_plan)
builder.add_node("Loan Payment", calculate_loan_payment)
builder.add_node("Human in the Loop", human_in_the_loop)
builder.add_node("Fallback", fallback)

builder.set_entry_point("Intent Detection")

def get_next_node(state: FinanceState):
    intent = state.get("intent", "unknown")
    mapping = {
        "profile": "Collect User Data",
        "stock": "Stock Info",
        "log_expense": "Process Expenses",
        "budget": "Budget Summary",
        "advice": "Provide Advice",
        "savings_plan": "Savings Plan",
        "loan_calculation": "Loan Payment",
        "human_in_the_loop": "Human in the Loop",
    }
    return mapping.get(intent, "Fallback")

builder.add_conditional_edges("Intent Detection", get_next_node)

# connect terminal edges to END constant (from langgraph.graph)
terminal_nodes = [
    "Process Expenses", "Collect User Data", "Stock Info", "Budget Summary",
    "Provide Advice", "Savings Plan", "Loan Payment", "Human in the Loop", "Fallback"
]
for n in terminal_nodes:
    builder.add_edge(n, END)

graph = builder.compile()

# --- 4. STREAMLIT UI COMPONENTS AND LOGIC ---
def get_base64_of_bin_file(bin_file: str) -> str:
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        logger.warning(f"Image file not found: {bin_file}")
        return ""

def set_custom_css(base64_string: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Financial Assistant", layout="wide")

# Apply CSS background
image_path = "FinAssisv2.png"
base64_image_string = get_base64_of_bin_file(image_path)
if base64_image_string:
    set_custom_css(base64_image_string)

st.title("💰 Financial Assistant")

with st.sidebar:
    st.header("How to use")
    st.write("Examples:\n- 'My name is Alex, I'm 30, yearly income $75,000.'\n- 'Log: rent $1500, groceries $450.'\n- 'How long to save for a $350k house if I save $500/month?'")

# initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = {}
if 'short_term_memory' not in st.session_state:
    st.session_state['short_term_memory'] = {}
if 'long_term_memory' not in st.session_state:
    st.session_state['long_term_memory'] = {}

# display existing messages
for msg in st.session_state['messages']:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# main chat input
if prompt := st.chat_input("Enter your request:"):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_state: FinanceState = {
        "user_input": prompt,
        "intent": "",
        "data": {},
        "user_profile": st.session_state['user_profile'],
        "short_term_memory": st.session_state['short_term_memory'],
        "long_term_memory": st.session_state['long_term_memory']
    }

    try:
        # graph.ainvoke is async; run it
        result = asyncio.run(graph.ainvoke(initial_state))

        # update session state
        st.session_state['user_profile'] = result.get('user_profile', st.session_state['user_profile'])
        st.session_state['short_term_memory'] = result.get('short_term_memory', st.session_state['short_term_memory'])
        st.session_state['long_term_memory'] = result.get('long_term_memory', st.session_state['long_term_memory'])

        assistant_response = result.get("data", {}).get("response", "Sorry, I couldn't produce a response.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during graph execution: {e}")
        assistant_response = "Sorry, an unexpected error occurred. Please try again."

    st.session_state['messages'].append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
