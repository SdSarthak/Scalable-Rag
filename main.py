from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from retrieval_chain import get_qa_chain
from auth import validate_token  # Implement Azure AD B2C validation
from prometheus_client import Counter, generate_latest
from ml_flow import log_query

app = FastAPI()
qa_chain = get_qa_chain()

REQUEST_COUNT = Counter("query_requests_total", "Total query requests")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest, token: str = Depends(validate_token)):
    result = qa_chain({"query": request.question})
    log_query(request.question, result["result"], result["source_documents"])
    REQUEST_COUNT.inc()
    return {"answer": result["result"], "sources": result["source_documents"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return generate_latest()