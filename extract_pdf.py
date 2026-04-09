import pdfplumber
import re


# ── Text Extraction ────────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ── Pattern helpers ────────────────────────────────────────────────────────────

def _find(pattern: str, text: str, flags=re.IGNORECASE) -> float | None:
    match = re.search(pattern, text, flags)
    if match:
        raw = match.group(1).replace(",", "").replace(" ", "")
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _find_two(pattern: str, text: str) -> tuple[float | None, float | None]:
    """Find a row with two numbers (current period + prior period)."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        def clean(s): return float(s.replace(",", "").replace(" ", ""))
        try:
            a = clean(match.group(1))
            b = clean(match.group(2)) if match.lastindex >= 2 else None
            return a, b
        except (ValueError, AttributeError):
            pass
    return None, None


NUM = r"([\d,]+(?:\.\d+)?)"   # matches numbers like 1,234,567 or 1234567.89


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_financials(text: str) -> dict:
    """
    Extract ALL financial fields from raw PDF text.
    Returns a dict with current-year fields AND prior-year fields prefixed with 'prior_'.
    """
    data = {}

    # ── Income Statement ──────────────────────────────────────────────────────

    # Revenue: try multiple label variations
    for label in [
        r"Net (?:turnover|revenue|sales)\s+" + NUM,
        r"Gross (?:turnover|revenue|sales)\s+" + NUM,
        r"(?:Total\s+)?Revenue\s+" + NUM,
    ]:
        cur, prior = _find_two(label.replace(NUM, f"{NUM}\\s+{NUM}"), text)
        if cur is None:
            cur = _find(label, text)
        if cur:
            data["revenue"] = cur
            if prior:
                data["prior_revenue"] = prior
            break

    # Gross profit
    cur, prior = _find_two(r"Gross profit\s+" + NUM + r"\s+" + NUM, text)
    if cur is None:
        cur = _find(r"Gross profit\s+" + NUM, text)
    if cur:
        data["gross_profit"] = cur
        if prior:
            data["prior_gross_profit"] = prior

    # Operating / EBIT
    cur, prior = _find_two(r"(?:Operating profit|Profit from operations|EBIT)\s+" + NUM + r"\s+" + NUM, text)
    if cur is None:
        cur = _find(r"(?:Operating profit|Profit from operations|EBIT)\s+" + NUM, text)
    if cur:
        data["operating_profit"] = cur
        if prior:
            data["prior_operating_profit"] = prior

    # PBT
    cur, prior = _find_two(r"Profit before (?:tax|taxation)\s+" + NUM + r"\s+" + NUM, text)
    if cur is None:
        cur = _find(r"Profit before (?:tax|taxation)\s+" + NUM, text)
    if cur:
        data["profit_before_tax"] = cur
        if prior:
            data["prior_profit_before_tax"] = prior

    # Net profit (PAT)
    for label in [
        r"Profit (?:for the (?:period|year)|after tax)\s+" + NUM + r"\s+" + NUM,
        r"Net profit\s+" + NUM + r"\s+" + NUM,
    ]:
        cur, prior = _find_two(label, text)
        if cur:
            data["net_profit"] = cur
            if prior:
                data["prior_net_profit"] = prior
            break
    if "net_profit" not in data:
        for label in [r"Profit (?:for the (?:period|year)|after tax)\s+" + NUM, r"Net profit\s+" + NUM]:
            v = _find(label, text)
            if v:
                data["net_profit"] = v
                break

    # EPS
    cur, prior = _find_two(r"(?:Earnings|EPS|Profit) per share\s+" + NUM + r"\s+" + NUM, text)
    if cur is None:
        cur = _find(r"(?:Earnings|EPS|Profit) per share[^\d]+" + NUM, text)
    if cur:
        data["eps"] = cur
        if prior:
            data["prior_eps"] = prior

    # Dividend per share
    v = _find(r"(?:Dividend|DPS) per share[^\d]+" + NUM, text)
    if v:
        data["dividend_per_share"] = v

    # Cost items
    for key, patterns in {
        "cost_of_goods_sold": [r"Cost of (?:goods sold|sales|revenue)\s+" + NUM],
        "operating_expenses":  [r"(?:Operating|Distribution|Admin(?:istrative)?) expenses\s+" + NUM],
        "depreciation":        [r"Depreciation\s+" + NUM, r"Depreciation and amortisation\s+" + NUM],
        "finance_cost":        [r"Finance costs?\s+" + NUM, r"Interest expense\s+" + NUM],
        "tax_expense":         [r"(?:Income )?[Tt]ax(?:ation)? expense\s+" + NUM, r"Provision for taxation\s+" + NUM],
    }.items():
        for pat in patterns:
            cur, prior = _find_two(pat.replace(NUM, f"{NUM}\\s+{NUM}"), text)
            if cur is None:
                cur = _find(pat, text)
            if cur:
                data[key] = cur
                prior_key = f"prior_{key}"
                if prior:
                    data[prior_key] = prior
                break

    # ── Balance Sheet ─────────────────────────────────────────────────────────

    for key, patterns in {
        "total_assets":            [r"Total assets\s+" + NUM],
        "current_assets":          [r"Total current assets\s+" + NUM, r"Current assets\s+" + NUM],
        "non_current_assets":      [r"Total non[- ]?current assets\s+" + NUM, r"Non[- ]?current assets\s+" + NUM],
        "cash_balance":            [r"Cash and (?:bank )?balances?\s+" + NUM, r"Cash and cash equivalents\s+" + NUM],
        "trade_receivables":       [r"Trade (?:and other )?receivables\s+" + NUM, r"Trade debtors\s+" + NUM],
        "inventory":               [r"Inventories\s+" + NUM, r"Stock in trade\s+" + NUM],
        "total_liabilities":       [r"Total liabilities\s+" + NUM],
        "current_liabilities":     [r"Total current liabilities\s+" + NUM, r"Current liabilities\s+" + NUM],
        "non_current_liabilities": [r"Total non[- ]?current liabilities\s+" + NUM],
        "total_equity":            [r"Total equity\s+" + NUM, r"(?:Shareholders'?|Owners') equity\s+" + NUM],
        "share_capital":           [r"Share capital\s+" + NUM, r"Paid[- ]?up (?:share )?capital\s+" + NUM],
        "long_term_debt":          [r"Long[- ]?term (?:debt|borrowings?|financing)\s+" + NUM],
    }.items():
        for pat in patterns:
            cur, prior = _find_two(pat.replace(NUM, f"{NUM}\\s+{NUM}"), text)
            if cur is None:
                cur = _find(pat, text)
            if cur:
                data[key] = cur
                if prior:
                    data[f"prior_{key}"] = prior
                break

    # ── Cash Flow ─────────────────────────────────────────────────────────────

    for key, patterns in {
        "operating_cashflow":  [r"Net cash (?:generated from|used in) operating activities\s+" + NUM],
        "investing_cashflow":  [r"Net cash (?:used in|from) investing activities\s+" + NUM],
        "financing_cashflow":  [r"Net cash (?:used in|from) financing activities\s+" + NUM],
    }.items():
        for pat in patterns:
            v = _find(pat, text)
            if v:
                data[key] = v
                break

    # ── Metadata ──────────────────────────────────────────────────────────────

    # Company name
    company_match = re.search(r"^([A-Z][A-Za-z\s]+(?:Limited|Ltd|Company|Corp|PLC))", text, re.MULTILINE)
    if company_match:
        data["_company"] = company_match.group(1).strip()

    # Reporting period
    period_match = re.search(
        r"(?:for the (?:period|year|quarter)|ended?)\s+(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{4}|Q[1-4]\s+\d{4})",
        text, re.IGNORECASE
    )
    if period_match:
        data["_period"] = period_match.group(1).strip()

    return data


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from db_insert import insert_financials

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/Test PDF.pdf"

    print(f"📄 Extracting from: {pdf_path}")
    text = extract_text(pdf_path)
    data = extract_financials(text)

    print("\n📊 Extracted fields:")
    for k, v in sorted(data.items()):
        if not k.startswith("_"):
            print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")

    company = data.get("_company", "Bestway Cement")
    period  = data.get("_period",  "FY 2025")
    year    = int(re.search(r"\d{4}", period).group()) if re.search(r"\d{4}", period) else 2025

    # Separate current and prior-year records
    current = {k: v for k, v in data.items() if not k.startswith("prior_") and not k.startswith("_")}
    prior   = {k.replace("prior_", ""): v for k, v in data.items() if k.startswith("prior_")}

    insert_financials(current, company=company, year=year, period=period)

    if prior:
        insert_financials(prior, company=company, year=year - 1, period=f"FY {year - 1}")

    print("\n✅ Done.")