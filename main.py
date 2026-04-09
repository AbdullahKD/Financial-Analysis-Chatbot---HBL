"""
main.py  —  FastAPI backend for Financial Analysis Chatbot
"""

import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from extract_pdf import extract_text, extract_financials
from db_insert import insert_financials
from db_query import run_query, get_all_years
from LLM_SQL import answer, classify_level, LEVEL_LABELS, is_detail_request

app = FastAPI(title="HBL Financial Analyst", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str   # "user" or "analyst"
    content: str


class ChatRequest(BaseModel):
    question: str
    company: str = "Bestway Cement"
    history: Optional[List[HistoryMessage]] = []


class ChatResponse(BaseModel):
    answer: str
    level: int
    level_label: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "HBL Financial Analyst v2"}


@app.get("/companies")
def get_companies():
    result = run_query("SELECT DISTINCT company, year, period FROM financials ORDER BY company, year DESC")
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {
        "companies": [
            {"company": r[0], "year": r[1], "period": r[2]}
            for r in result["rows"]
        ]
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), company: str = "Bestway Cement"):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path)
        data = extract_financials(text)

        if not data:
            raise HTTPException(status_code=422, detail="No financial data could be extracted from this PDF.")

        # Use company name from UI, not from PDF (PDF titles are messy)
        # company = data.get("_company", company)
        period  = data.get("_period", "FY 2025")
        year    = int(re.search(r"\d{4}", period).group()) if re.search(r"\d{4}", period) else 2025

        current = {k: v for k, v in data.items() if not k.startswith("prior_") and not k.startswith("_")}
        prior   = {k.replace("prior_", ""): v for k, v in data.items() if k.startswith("prior_")}

        insert_financials(current, company=company, year=year, period=period)

        prior_inserted = False
        if prior:
            insert_financials(prior, company=company, year=year - 1, period=f"FY {year - 1}")
            prior_inserted = True

        return {
            "status": "success",
            "company": company,
            "period": period,
            "fields_extracted": len(current),
            "prior_year_extracted": prior_inserted,
            "extracted_fields": list(current.keys()),
        }

    finally:
        os.unlink(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Build history string from last few exchanges
    history = ""
    if req.history:
        for msg in req.history[-6:]:  # last 3 exchanges (6 messages)
            role = "User" if msg.role == "user" else "Analyst"
            history += f"{role}: {msg.content}\n\n"

    level = 0 if (is_detail_request(req.question) and history) else classify_level(req.question)
    level_label = "Detail Follow-up" if level == 0 else LEVEL_LABELS[level]

    response = answer(req.question, req.company, history)

    return ChatResponse(
        answer=response,
        level=level if level != 0 else 4,
        level_label=level_label,
        question=req.question,
    )


if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)