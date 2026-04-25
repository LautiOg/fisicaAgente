import os
import json
import shutil
import time
import base64
import re
import traceback
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# ── Configuración ──────────────────────────────────────────────────────────────
DOCS_DIR      = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR    = os.path.join(os.path.dirname(__file__), "chroma_db_google") 
INDEXED_FILE  = os.path.join(os.path.dirname(__file__), "indexed_files_google.json")

LLM_MODEL = "gemini-flash-latest"

# Cargar múltiples API Keys si existen
API_KEYS = os.getenv("GOOGLE_API_KEY", "").split(",")
current_key_index = 0

# Usar la primera llave para los embeddings
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=API_KEYS[0].strip()
)

def _get_vectorstore():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)

def ingest_new_pdfs():
    indexed = _load_indexed(); all_pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    new_pdfs = [f for f in all_pdfs if f not in indexed]
    if not new_pdfs: return
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for f in new_pdfs:
        try:
            loader = PyPDFLoader(os.path.join(DOCS_DIR, f)); pages = loader.load(); chunks = splitter.split_documents(pages)
            Chroma.from_documents(documents=chunks, embedding=EMBEDDINGS, persist_directory=CHROMA_DIR)
            indexed.add(f)
        except: continue
    _save_indexed(indexed)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest_new_pdfs()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None

@app.post("/chat")
def chat(request: ChatRequest):
    global current_key_index
    try:
        vectorstore = _get_vectorstore()
        docs = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(request.message)
        context = "\n\n".join(d.page_content for d in docs)

        system_prompt = f"""Eres un asistente de física universitario. Respondés ÚNICA Y EXCLUSIVAMENTE basándote en el CONTEXTO BIBLIOGRÁFICO provisto más abajo.

═══ REGLAS DE FIDELIDAD (OBLIGATORIAS) ═══
1. SOLO usá información del CONTEXTO. Si el tema no aparece ahí, respondé exactamente: "Esa información no está contemplada en el material disponible."
2. Podés deducir y resolver matemáticamente a partir de los conceptos que SÍ están en el contexto.
3. Citá siempre el documento fuente (nombre y página) al explicar.

═══ VISUALIZACIONES ═══

CASO A — Cinemática lineal (MRU/MRUV): generá OBLIGATORIAMENTE 3 bloques [CHART]:
[CHART]
{{
  "title": "Posición vs Tiempo",
  "xAxis": "Tiempo (s)",
  "yAxis": "Posición (m)",
  "series": [
    {{"name": "Objeto 1", "data": [{{"x": 0, "y": 0}}, {{"x": 5, "y": 10}}, {{"x": 10, "y": 20}}]}},
    {{"name": "Objeto 2", "data": [{{"x": 0, "y": 0}}, {{"x": 5, "y": 2.5}}, {{"x": 10, "y": 10}}]}}
  ]
}}
[/CHART]
(Repetir estructura para Velocidad y Aceleración. Los valores de x e y deben ser SOLO números, sin unidades.)

CASO B — Polares / Intrínsecas / Vectores: generá bloques [DIAGRAM]:
[DIAGRAM]
{{
  "title": "Diagrama vectorial",
  "zoom": 1,
  "elements": [
    {{"type": "circle", "r": 2}},
    {{"type": "point", "x": 1.41, "y": 1.41, "label": "A", "color": "#5865F2"}},
    {{"type": "vector", "x": 1.41, "y": 1.41, "vx": -0.7, "vy": 0.7, "label": "v_A", "color": "#4ADE80"}},
    {{"type": "versor", "x": 1.41, "y": 1.41, "vx": 0.7, "vy": 0.7, "label": "e_r"}}
  ]
}}
[/DIAGRAM]
(x e y se calculan con x=r·cos(θ), y=r·sin(θ). Versores parten de la partícula.)

═══ RESOLUCIÓN ═══
- Explicá PASO A PASO con rigor académico.
- Usá LaTeX ($...$) para todas las fórmulas.

═══ CONTEXTO BIBLIOGRÁFICO ═══
{context}
"""
        content = [{"type": "text", "text": request.message}]
        if request.image: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image}"}})

        last_error = ""
        for _ in range(len(API_KEYS)):
            try:
                key = API_KEYS[current_key_index].strip()
                llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=key, temperature=0.1)
                raw = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)]).content
                if isinstance(raw, list): raw = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in raw])

                charts = []; diagrams = []; clean = raw
                for m in re.findall(r"\[CHART\](.*?)\[/CHART\]", raw, re.DOTALL):
                    try: charts.append(json.loads(m.strip())); clean = clean.replace(f"[CHART]{m}[/CHART]", "")
                    except: pass
                for m in re.findall(r"\[DIAGRAM\](.*?)\[/DIAGRAM\]", raw, re.DOTALL):
                    try: diagrams.append(json.loads(m.strip())); clean = clean.replace(f"[DIAGRAM]{m}[/DIAGRAM]", "")
                    except: pass

                sources = [{"page": d.metadata.get("page", "?"), "source": os.path.basename(d.metadata.get("source", "?"))} for d in docs]
                return {"response": clean.strip(), "sources": sources, "charts": charts, "diagrams": diagrams}

            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "403" in last_error or "PERMISSION_DENIED" in last_error:
                    print(f"⚠️ Cambiando API Key...")
                    current_key_index = (current_key_index + 1) % len(API_KEYS)
                    continue
                else: raise e

        raise HTTPException(status_code=500, detail=f"Agotado: {last_error}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def docs(): return {"indexed": list(_load_indexed())}
