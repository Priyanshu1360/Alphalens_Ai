# ============================================================
#  Alphalens AI -- Kaggle Ingestion Script
#  Run this as a Kaggle Notebook (Python script)
# ============================================================
#
#  SETUP STEPS:
#  1. Create a new Kaggle Notebook
#  2. Upload your 28 PDFs as a Kaggle Dataset (flat folder)
#  3. In notebook: Add Data -> attach your dataset
#  4. Copy-paste this entire script into one code cell
#  5. Set PDF_DATASET_PATH below to your dataset path
#  6. Enable GPU (optional, faster embedding)
#  7. Run All -> wait ~10-15 min
#  8. Output tab -> Download bm25_sparse_stats.json
#
#  AFTER DOWNLOAD:
#  - Copy bm25_sparse_stats.json -> c:/my_rag/my_rag/Data/
#  - Update your local .env with the Qdrant Cloud URL and API Key
# ============================================================

import subprocess, sys

# STEP 1: Install packages
print("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "python-dotenv",
                "docling",
                "pandas",
                "tabulate",
                "langchain-text-splitters",
                "sentence-transformers",
                "qdrant-client>=1.12.0"], check=False)
print("Done.\n")

# STEP 2: Configuration
import os, re, json, math, hashlib, zipfile
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# UPDATE THIS to your Kaggle dataset path
# After attaching dataset it appears at /kaggle/input/<dataset-slug>/
PDF_DATASET_PATH   = "/kaggle/input/alphalens-financial-pdfs"   # <-- CHANGE THIS

# Qdrant Cloud Config
QDRANT_URL         = os.getenv("QDRANT_URL", "YOUR_QDRANT_CLOUD_URL")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY", "YOUR_QDRANT_CLOUD_API_KEY")

BM25_STATS_PATH    = "/kaggle/working/bm25_sparse_stats.json"

EMBEDDING_MODEL    = "BAAI/bge-large-en-v1.5"
QDRANT_COLLECTION  = "finance_rag_v2"
DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

TEXT_CHUNK_SIZE    = 300
TEXT_CHUNK_OVERLAP = 50
TABLE_CHUNK_SIZE   = 120
TABLE_CHUNK_OVERLAP= 20
EMBED_BATCH_SIZE   = 64
SPARSE_HASH_SPACE  = 2_000_003
BM25_K1            = 1.5
BM25_B             = 0.75
QUALITY_HIGH       = 0.75
QUALITY_MEDIUM     = 0.55
UPSERT_BATCH       = 100

# os.makedirs(QDRANT_OUTPUT_PATH, exist_ok=True)

if not os.path.exists(PDF_DATASET_PATH):
    raise FileNotFoundError(f"PDF dataset not found: {PDF_DATASET_PATH}\nPlease attach your dataset and update PDF_DATASET_PATH.")

pdfs = sorted(Path(PDF_DATASET_PATH).rglob("*.pdf"))
print(f"Found {len(pdfs)} PDFs:")
for p in pdfs: print(f"  {p.name}")
print()

# STEP 3: PDF Loader
import pandas as pd
from docling.document_converter import DocumentConverter

YEAR_PAT    = re.compile(r"\b(20\d{2})\b")
QUARTER_PAT = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
COMPANIES   = ["amazon","apple","google","meta","microsoft"]

def extract_meta(filename):
    low = filename.lower()
    return {
        "company":     next((c for c in COMPANIES if c in low), "unknown"),
        "report_type": ("10-k" if "10-k" in low else "10-q" if "10-q" in low else "8-k" if "8-k" in low else None),
        "year":        (m := YEAR_PAT.search(low)) and m.group(1),
        "quarter":     (m := QUARTER_PAT.search(low)) and m.group(1),
    }

def load_pdfs(pdf_paths):
    converter = DocumentConverter()
    docs = []
    for path in pdf_paths:
        try:
            res = converter.convert(str(path))
            doc = res.document
            
            # Extract entire document text as markdown
            text_md = doc.export_to_markdown()
            
            # Extract tables individually for special chunking
            tables = []
            for tbl in doc.tables:
                try:
                    df = tbl.export_to_dataframe()
                    tables.append(df.to_markdown(index=False))
                except:
                    pass
                    
            meta = extract_meta(path.name)
            docs.append({"text": text_md, "tables": tables,
                         "file_name": path.name, "file_path": str(path), **meta})
            print(f"  Loaded: {path.name}")
        except Exception as e:
            print(f"  WARNING: Skipped {path.name}: {e}")
    return docs

print("Loading PDFs...")
docs = load_pdfs(pdfs)
print(f"\nLoaded: {len(docs)} documents\n")

# STEP 4: Chunking
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def create_chunks(docs):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Character splitter for sections that are still too long (approx 4 chars per word)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE * 4, 
        chunk_overlap=TEXT_CHUNK_OVERLAP * 4
    )
    
    # Table splitter is just naive sliding window for now since tables are dense strings
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE * 4, 
        chunk_overlap=TABLE_CHUNK_OVERLAP * 4
    )

    all_chunks = []
    for doc in docs:
        base_meta = {k: doc.get(k) for k in ["company","report_type","year","quarter","file_name","file_path"]}
        
        # 1. Process Markdown Text
        md_text = doc.get("text", "")
        if md_text:
            # Split by headers first
            header_splits = md_splitter.split_text(md_text)
            
            idx = 1
            for split in header_splits:
                # If a semantic header section is STILL too long, split it by size
                size_splits = text_splitter.split_text(split.page_content)
                
                # Combine base metadata with any headers detected by the markdown splitter
                chunk_meta = {**base_meta, **split.metadata}
                
                for s in size_splits:
                    all_chunks.append({
                        "text": s, 
                        "chunk_type": "text", 
                        "chunk_index": idx, 
                        **chunk_meta
                    })
                    idx += 1
                    
        # 2. Process Tables (Keep them separate for structural coherence scoring)
        for ti, tbl in enumerate(doc.get("tables",[]), 1):
            table_splits = table_splitter.split_text(tbl)
            for pi, c in enumerate(table_splits, 1):
                all_chunks.append({
                    "text": c, 
                    "chunk_type": "table", 
                    "chunk_index": pi, 
                    "table_index": ti, 
                    **base_meta
                })
    return all_chunks

print("Chunking...")
chunks = create_chunks(docs)
print(f"Chunks: {len(chunks)} (text={sum(1 for c in chunks if c['chunk_type']=='text')}, table={sum(1 for c in chunks if c['chunk_type']=='table')})\n")

# STEP 5: Clean + Coherence
_WRE = re.compile(r"[A-Za-z0-9]+")
def _tok(t): return _WRE.findall((t or "").lower())
def _clamp(v): return max(0.0, min(1.0, v))

def text_coh(text):
    w = _tok(text); wc = len(w)
    if wc == 0: return 0.0
    ls = _clamp(wc/80.0); ur = len(set(w))/wc; ds = _clamp(1.0 - abs(ur-0.55)/0.55)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents)>=2:
        sims = []
        for i in range(len(sents)-1):
            a,b = set(_tok(sents[i])),set(_tok(sents[i+1]))
            sims.append(len(a&b)/len(a|b) if a|b else 0)
        cs = _clamp((sum(sims)/len(sims))*4.0)
    else: cs = 0.5
    return round(_clamp(0.35*ls+0.35*ds+0.30*cs),4)

def table_coh(text):
    w = _tok(text); wc = len(w)
    if wc==0: return 0.0
    ls = _clamp(wc/60.0)
    lines = [l for l in text.splitlines() if l.strip()]
    tl = [l for l in lines if "|" in l]
    ss = _clamp(sum(1 for c in [l.count("|")+1 for l in tl] if c==(([l.count("|")+1 for l in tl])[0]))/len(tl)) if tl else 0.4
    ur = len(set(w))/wc; ds = _clamp(1.0-abs(ur-0.65)/0.65)
    return round(_clamp(0.40*ls+0.40*ss+0.20*ds),4)

def qlabel(s): return "high" if s>=QUALITY_HIGH else "medium" if s>=QUALITY_MEDIUM else "low"

def clean_chunks(chunks):
    out = []
    for c in chunks:
        t = " ".join((c.get("text") or "").split())
        if not t: continue
        cc = dict(c); cc["text"] = t
        score = table_coh(t) if c.get("chunk_type")=="table" else text_coh(t)
        cc["coherence_score"] = score; cc["chunk_quality"] = qlabel(score)
        out.append(cc)
    return out

print("Cleaning...")
cleaned = clean_chunks(chunks)
print(f"After cleaning: {len(cleaned)}, quality={dict(Counter(c['chunk_quality'] for c in cleaned))}\n")

# STEP 6: BM25 Stats
def _hidx(t): return int.from_bytes(hashlib.sha1(t.encode()).digest()[:4],"big") % SPARSE_HASH_SPACE

def build_bm25(texts):
    dc, tl = 0, 0; idf = Counter()
    for text in texts:
        toks = _tok(text)
        if not toks: continue
        dc+=1; tl+=len(toks); seen=set()
        for t in toks:
            idx=_hidx(t)
            if idx not in seen: seen.add(idx); idf[idx]+=1
    return {"doc_count": dc, "avg_doc_len": tl/dc if dc else 0.0, "index_doc_freq": {str(k):v for k,v in idf.items()}}

print("Building BM25 stats...")
texts = [c["text"] for c in cleaned]
stats = build_bm25(texts)
with open(BM25_STATS_PATH,"w") as f: json.dump(stats,f)
print(f"BM25: doc_count={stats['doc_count']}, avg_len={stats['avg_doc_len']:.1f}\n")

# STEP 7: Dense Embeddings
from sentence_transformers import SentenceTransformer
import torch

print(f"Loading model: {EMBEDDING_MODEL} (using PyTorch GPU)...")
# PyTorch is natively pre-configured on Kaggle to use the GPU perfectly
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print(f"Model ready on {device}.\n")

def embed_all(texts):
    print(f"Embedding {len(texts)} chunks...")
    # sentence-transformers handles batching internally and shows a progress bar automatically
    embeddings = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True)
    return embeddings.tolist()

embeddings = embed_all(texts)
VDIM = len(embeddings[0])
print(f"Done. dim={VDIM}\n")

# STEP 8: Sparse Vectors
from qdrant_client.models import SparseVector
def _bm25w(tf,df,N,dl,adl):
    if tf<=0 or df<=0 or N<=0: return 0.0
    adl=max(float(adl),1e-9)
    idf_v=math.log(1.0+((N-df+0.5)/(df+0.5)))
    den=tf+BM25_K1*(1.0-BM25_B+BM25_B*(dl/adl))
    return idf_v*((tf*(BM25_K1+1.0))/den) if den>0 else 0.0

def sparse_vec(text, stats):
    toks=_tok(text)
    if not toks: return SparseVector(indices=[],values=[])
    cnt=Counter(toks); N=int(stats.get("doc_count",0)); adl=float(stats.get("avg_doc_len",0)); dl=len(toks)
    imap={int(k):int(v) for k,v in stats.get("index_doc_freq",{}).items()}
    h={}
    for t,c in cnt.items():
        idx=_hidx(t); w=_bm25w(c,imap.get(idx,0),N,dl,adl)
        if w>0: h[idx]=h.get(idx,0.0)+w
    items=sorted(h.items())
    return SparseVector(indices=[i for i,_ in items],values=[round(v,6) for _,v in items])

print("Building sparse vectors...")
svecs = [sparse_vec(c["text"], stats) for c in cleaned]
print(f"Done. {len(svecs)} sparse vectors\n")

# STEP 9: Store in Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)
existing = {c.name for c in client.get_collections().collections}
if QDRANT_COLLECTION in existing:
    print(f"Deleting old collection '{QDRANT_COLLECTION}'...")
    client.delete_collection(QDRANT_COLLECTION)

client.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config={DENSE_VECTOR_NAME: VectorParams(size=VDIM, distance=Distance.COSINE)},
    sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
)
print(f"Collection '{QDRANT_COLLECTION}' created (dim={VDIM})\n")

def pid(chunk):
    raw="|".join(str(chunk.get(k,"")) for k in ["file_path","chunk_type","chunk_index","table_index","text"])
    return int(hashlib.sha1(raw.encode()).hexdigest()[:16],16)

inserted=0
for start in range(0,len(cleaned),UPSERT_BATCH):
    pts=[PointStruct(id=pid(c),vector={DENSE_VECTOR_NAME:d,SPARSE_VECTOR_NAME:s},payload=c)
         for c,d,s in zip(cleaned[start:start+UPSERT_BATCH],embeddings[start:start+UPSERT_BATCH],svecs[start:start+UPSERT_BATCH])]
    client.upsert(collection_name=QDRANT_COLLECTION,points=pts)
    inserted+=len(pts); print(f"  Upserted {inserted}/{len(cleaned)}...")

final=client.count(QDRANT_COLLECTION).count
print(f"\nIngestion complete! Total vectors: {final}")
client.close()

# STEP 10: Zip & Export
print("\n" + "="*55)
print("DONE! Vectors have been pushed to your Qdrant Cloud cluster.")
print("Next steps:")
print("1. Kaggle -> Output tab -> Download bm25_sparse_stats.json")
print("2. Copy bm25_sparse_stats.json -> c:/my_rag/my_rag/Data/")
print("3. Ensure your local .env has the same QDRANT_URL and QDRANT_API_KEY")
print("4. Run: .\\run_fullstack.ps1")
print("="*55)
