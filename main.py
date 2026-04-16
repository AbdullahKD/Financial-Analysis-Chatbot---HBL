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

from extract_pdf import extract_text_only as extract_text, extract_financials, extract_financials_intelligent
from db_insert import insert_financials
from db_query import run_query, get_all_years
from LLM_SQL import answer, store_pdf_text, invalidate_cache, LEVEL_LABELS

app = FastAPI(title="HBL Financial Analyst", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str
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
        # Phase 1-4: Intelligent extraction with Mistral + validation
        result = extract_financials_intelligent(tmp_path)

        current  = result["current"]
        prior    = result["prior"]
        meta     = result["metadata"]
        raw_text = result["raw_text"]

        if not current:
            raise HTTPException(status_code=422, detail="No financial data could be extracted. Check that the PDF contains financial statements.")

        # Store full PDF text for qualitative Q&A
        store_pdf_text(company, raw_text)
        # Invalidate context cache so next query gets fresh data
        invalidate_cache(company)

        # Determine year from extracted period
        period_str = meta.get("period_current") or "31 December 2025"
        year_match = re.search(r"\d{4}", period_str)
        year = int(year_match.group()) if year_match else 2025
        period_label = f"H1 FY{year}"

        insert_financials(current, company=company, year=year, period=period_label)

        prior_inserted = False
        if prior:
            prior_str = meta.get("period_prior") or str(year - 1)
            prior_year_match = re.search(r"\d{4}", prior_str)
            prior_year = int(prior_year_match.group()) if prior_year_match else year - 1
            insert_financials(prior, company=company, year=prior_year, period=f"H1 FY{prior_year}")
            prior_inserted = True

        # Build validation summary for response
        validation = meta.get("validation", {})
        passed = len(validation.get("passed", []))
        failed = len(validation.get("failed", []))

        return {
            "status": "success",
            "company": company,
            "period": period_label,
            "fields_extracted": len(current),
            "prior_year_extracted": prior_inserted,
            "extracted_fields": list(current.keys()),
            "validation": {
                "checks_passed": passed,
                "checks_failed": failed,
                "warnings": validation.get("warnings", []),
                "failed_checks": validation.get("failed", [])
            }
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
        for msg in req.history[-6:]:
            role = "User" if msg.role == "user" else "Analyst"
            history += f"{role}: {msg.content}\n\n"

    response_text, level, level_label = answer(req.question, req.company, history)

    return ChatResponse(
        answer=response_text,
        level=level,
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