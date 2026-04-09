"""
LLM_SQL.py  —  Hybrid financial Q&A engine
"""

import subprocess
import re
from db_query import run_query, get_financial_context, get_all_years, format_result


def ask_llm(prompt: str, system: str = "") -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    result = subprocess.run(
        ["ollama", "run", "llama3:8b"],
        input=full_prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode().strip()


LEVEL_KEYWORDS = {
    1: [
        "what is", "what are", "how much", "company name", "reporting period",
        "total revenue", "total assets", "total liabilities", "share capital",
        "cash", "finance cost", "eps", "earnings per share", "profit after tax",
        "net profit", "turnover"
    ],
    2: [
        "compare", "compared", "last year", "prior year", "year-over-year",
        "changed", "change", "increase", "decrease", "grew", "fell",
        "higher", "lower", "more than", "less than", "between 20"
    ],
    3: [
        "margin", "ratio", "calculate", "return on", "roa", "roe",
        "debt-to-equity", "current ratio", "quick ratio", "asset turnover",
        "operating margin", "net profit margin", "gross profit margin",
        "eps growth", "percentage of revenue"
    ],
    4: [
        "why", "despite", "contributors", "major", "leveraged", "liquidity",
        "efficiently", "trend", "sustainable", "dependent", "risks",
        "worsening", "improving"
    ],
    5: [
        "attractive", "investor", "healthy", "strengths", "weaknesses",
        "concerned", "dividend policy", "overvalued", "undervalued",
        "growth-oriented", "mature", "capital efficiently", "key metrics",
        "would you invest", "management"
    ],
}

DETAIL_KEYWORDS = ["more detail", "elaborate", "explain more", "expand", "go deeper", "tell me more", "provide more"]


def classify_level(question: str) -> int:
    q = question.lower()
    for level in [5, 4, 3, 2, 1]:
        if any(kw in q for kw in LEVEL_KEYWORDS[level]):
            return level
    return 1


def is_detail_request(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DETAIL_KEYWORDS)


FULL_SCHEMA = """
Table: financials
Columns: id, company, year, period,
  revenue, gross_profit, operating_profit, profit_before_tax, net_profit,
  eps, dividend_per_share,
  cost_of_goods_sold, operating_expenses, depreciation, finance_cost, tax_expense,
  total_assets, current_assets, non_current_assets, cash_balance,
  trade_receivables, inventory,
  total_liabilities, current_liabilities, non_current_liabilities,
  total_equity, share_capital, long_term_debt,
  operating_cashflow, investing_cashflow, financing_cashflow
"""


def generate_sql(question: str, company: str = "Bestway Cement") -> str:
    years = get_all_years(company)
    latest_year = years[0] if years else 2025

    prompt = f"""You are a PostgreSQL expert converting finance questions to SQL.

{FULL_SCHEMA}

RULES:
- Return ONLY executable SQL, no explanation, no markdown, no backticks
- Company is '{company}' unless stated otherwise
- Latest available year is {latest_year}
- For comparisons, SELECT both years in one query
- Use ORDER BY year DESC for multi-year queries
- If question asks for ratio/calculation, SELECT the raw columns needed (not the calculation)

Question: {question}

SQL:"""
    raw = ask_llm(prompt)
    return clean_sql(raw)


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()
    match = re.search(r"(SELECT\s.+?)(?:;|$)", sql, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip() + ";"
    return sql


SYSTEM_ANALYST = """You are a senior financial analyst. Be concise and professional.
Default to brief answers (3-5 lines max). Only elaborate if the user explicitly asks for detail.
Use numbers to support every point. No filler phrases."""

def handle_l1(question: str, company: str, history: str = "") -> str:
    years = get_all_years(company)

    if not years:
        return "No data found for this company."

    latest_year = years[0]

    # Map user keywords to database columns
    metric_map = {
        "revenue": "revenue",
        "sales": "revenue",
        "turnover": "revenue",

        "gross profit": "gross_profit",
        "operating profit": "operating_profit",

        "profit before tax": "profit_before_tax",
        "pbt": "profit_before_tax",

        "net profit": "net_profit",
        "profit after tax": "net_profit",
        "pat": "net_profit",

        "eps": "eps",
        "earnings per share": "eps",

        "assets": "total_assets",
        "total assets": "total_assets",

        "liabilities": "total_liabilities",
        "total liabilities": "total_liabilities",

        "cash": "cash_balance",
        "cash balance": "cash_balance",

        "share capital": "share_capital",

        "finance cost": "finance_cost",
        "finance costs": "finance_cost"
    }

    question_lower = question.lower()

    selected_metric = None

    for keyword, column in metric_map.items():
        if keyword in question_lower:
            selected_metric = column
            break

    if not selected_metric:
        return "Sorry, I couldn't determine what financial metric you're asking for."

    query = f"""
        SELECT year, period, {selected_metric}
        FROM financials
        WHERE company = '{company}'
        AND year = {latest_year}
        LIMIT 1
    """

    result = run_query(query)

    if "error" in result or not result["rows"]:
        return "No data found."

    row = result["rows"][0]

    year = row[0]
    period = row[1]
    value = row[2]

    metric_name = selected_metric.replace("_", " ").title()

    return f"{company}'s {metric_name} for {period} was PKR {value:,.0f}."

def handle_l2(question: str, company: str, history: str = "") -> str:
    sql = generate_sql(question, company)
    result = run_query(sql)
    if "error" in result:
        return f"⚠️ SQL Error: {result['error']}"
    data_text = format_result(result)

    prompt = f"""Data:
{data_text}

Question: {question}

Give a 2-3 line answer with specific numbers and the percentage change. State which year each figure is from. No preamble.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l3(question: str, company: str, history: str = "") -> str:
    context = get_financial_context(company)

    prompt = f"""Financial data:
{context}

Question: {question}

Respond in this exact format:
Formula: [formula]
Values: [extracted values with years]
Calculation: [one-line calc]
Result: [answer with %/ratio]
Interpretation: [1 sentence only]

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l4(question: str, company: str, history: str = "") -> str:
    context = get_financial_context(company)

    prompt = f"""Financial data:
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer in 3-4 lines maximum. Lead with the conclusion, support with 1-2 key numbers. No bullet points unless asked.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_l5(question: str, company: str, history: str = "") -> str:
    context = get_financial_context(company)

    prompt = f"""Financial data:
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer in 4-5 lines maximum. Lead with a clear verdict, support with the most relevant metrics. Be direct.

End your response with exactly this line: "Need more detail? Just ask." """

    return ask_llm(prompt, system=SYSTEM_ANALYST)


def handle_detail(question: str, company: str, history: str = "") -> str:
    """Handles follow-up detail requests using previous conversation context."""
    context = get_financial_context(company)

    prompt = f"""Financial data:
{context}

Previous conversation:
{history}

The user is asking for more detail on the previous answer. Provide a thorough, well-structured expansion.
Use specific numbers. Use bullet points for clarity. Be professional.

User: {question}"""

    return ask_llm(prompt, system=SYSTEM_ANALYST)


LEVEL_HANDLERS = {
    1: handle_l1,
    2: handle_l2,
    3: handle_l3,
    4: handle_l4,
    5: handle_l5,
}

LEVEL_LABELS = {
    1: "Basic Retrieval",
    2: "Comparative",
    3: "Ratio Analysis",
    4: "Analytical Reasoning",
    5: "Investor Insight",
}


def answer(question: str, company: str = "Bestway Cement", history: str = "") -> str:
    if is_detail_request(question) and history:
        print("  [Detail follow-up]")
        return handle_detail(question, company, history)

    level = classify_level(question)
    handler = LEVEL_HANDLERS[level]
    print(f"  [Level {level} — {LEVEL_LABELS[level]}]")
    return handler(question, company, history)


if __name__ == "__main__":
    company = "Bestway Cement"
    print(f"💬 Financial Analyst — {company}")
    print("Type 'exit' to quit.\n")
    history = ""

    while True:
        question = input("❓ Your question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        print()
        response = answer(question, company, history)
        print(f"\n📊 Answer:\n{response}\n")
        print("-" * 60)

        # Update history (keep last 2 exchanges)
        history += f"User: {question}\nAnalyst: {response}\n\n"
        history_lines = history.strip().split("\n")
        if len(history_lines) > 12:
            history = "\n".join(history_lines[-12:])