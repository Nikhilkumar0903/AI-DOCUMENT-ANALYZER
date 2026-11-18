from __future__ import annotations
import asyncio, json, os, re, shutil, sqlite3, threading, uuid, zipfile, logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from .auth import current_user, local_auth_enabled

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
DATA = Path(os.getenv("DOCUMIND_DATA_DIR", str(BASE / "data"))).expanduser().resolve()
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
MAX_FILE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_FILES = int(os.getenv("MAX_UPLOAD_FILES", "10"))
MAX_TOTAL_UPLOAD_BYTES = int(
    os.getenv("MAX_UPLOAD_TOTAL_MB", str(MAX_FILE_BYTES * MAX_FILES // (1024 * 1024)))
) * 1024 * 1024
MAX_ANALYSES_PER_USER_PER_DAY = max(1, int(os.getenv("MAX_ANALYSES_PER_USER_PER_DAY", "3")))
JOB_RETENTION_DAYS = max(1, int(os.getenv("JOB_RETENTION_DAYS", "7")))
ALLOWED = {".pdf",".docx",".doc",".xlsx",".xls",".xlsm",".csv",".tsv",".txt",".md",".log",".json",".jsonl",".xml",".html",".htm",".css",".js",".ts",".py",".java",".c",".cpp",".h",".hpp",".sql",".yaml",".yml",".ini",".cfg",".conf",".sh",".bat",".ps1",".png",".jpg",".jpeg",".gif",".bmp",".tiff",".webp",".zip"}
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("document-analyzer")
_USAGE_DB = DATA / "usage.sqlite3"
if not local_auth_enabled() and os.getenv("AUTH_REQUIRED", "false").strip().lower() not in {
    "1",
    "true",
    "yes",
}:
    raise RuntimeError(
        "Authentication cannot be disabled on a non-local host. Set AUTH_REQUIRED=true."
    )


def _connect_usage_db():
    connection = sqlite3.connect(_USAGE_DB, timeout=30)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS daily_usage ("
        "user_id TEXT NOT NULL, usage_date TEXT NOT NULL, analyses INTEGER NOT NULL, "
        "upload_jobs INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (user_id, usage_date))"
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(daily_usage)")}
    if "upload_jobs" not in columns:
        connection.execute(
            "ALTER TABLE daily_usage ADD COLUMN upload_jobs INTEGER NOT NULL DEFAULT 0"
        )
    return connection


def _consume_daily_counter(user_id: str, column: str, limit: int, label: str) -> int:
    usage_date = datetime.now(timezone.utc).date().isoformat()
    connection = _connect_usage_db()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            f"SELECT {column} FROM daily_usage WHERE user_id=? AND usage_date=?",
            (user_id, usage_date),
        ).fetchone()
        used = int(row[0]) if row else 0
        if used >= limit:
            connection.rollback()
            raise HTTPException(
                429,
                f"Daily {label} limit reached ({limit} per account). Try again tomorrow.",
            )
        connection.execute(
            f"INSERT INTO daily_usage (user_id, usage_date, analyses, upload_jobs) "
            f"VALUES (?, ?, {'1' if column == 'analyses' else '0'}, "
            f"{'1' if column == 'upload_jobs' else '0'}) "
            f"ON CONFLICT(user_id, usage_date) DO UPDATE SET {column}={column}+1",
            (user_id, usage_date),
        )
        connection.commit()
        return used + 1
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def consume_daily_quota(user_id: str) -> int:
    return _consume_daily_counter(
        user_id, "analyses", MAX_ANALYSES_PER_USER_PER_DAY, "analysis"
    )


def consume_upload_quota(user_id: str) -> int:
    return _consume_daily_counter(
        user_id, "upload_jobs", MAX_ANALYSES_PER_USER_PER_DAY, "upload job"
    )


def _refund_daily_counter(user_id: str, column: str):
    usage_date = datetime.now(timezone.utc).date().isoformat()
    connection = _connect_usage_db()
    try:
        connection.execute(
            f"UPDATE daily_usage SET {column}=MAX(0, {column}-1) "
            "WHERE user_id=? AND usage_date=?",
            (user_id, usage_date),
        )
        connection.commit()
    finally:
        connection.close()


def _refund_daily_quota(user_id: str):
    _refund_daily_counter(user_id, "analyses")


def _refund_upload_quota(user_id: str):
    _refund_daily_counter(user_id, "upload_jobs")


def cleanup_expired_jobs() -> int:
    if local_auth_enabled():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=JOB_RETENTION_DAYS)
    removed = 0
    for folder in JOBS.iterdir():
        job_file = folder / "job.json"
        if not folder.is_dir() or not job_file.is_file():
            continue
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(job["created_at"]).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            log.exception("Unable to determine retention for job metadata: %s", job_file)
            continue
        if created_at < cutoff and job.get("status") not in ("queued", "processing"):
            try:
                shutil.rmtree(folder)
                removed += 1
            except OSError:
                log.exception("Unable to remove expired analysis job: %s", folder)
    if removed:
        log.info("Removed %s analysis job(s) older than %s days", removed, JOB_RETENTION_DAYS)
    return removed


async def _retention_loop():
    while True:
        await asyncio.sleep(3600)
        cleanup_expired_jobs()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    cleanup_expired_jobs()
    for job_file in JOBS.glob("*/job.json"):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("Unable to read job metadata during startup recovery: %s", job_file)
            continue
        if job.get("status") in ("queued", "processing"):
            job.update(
                status="failed",
                stage="Interrupted by server restart",
                error="The server restarted before this analysis finished. Retry the job to continue.",
                updated_at=now(),
            )
            write_job(job)
            log.warning("Marked interrupted analysis job as failed and retryable: %s", job.get("job_id"))
    retention_task = asyncio.create_task(_retention_loop())
    try:
        yield
    finally:
        retention_task.cancel()
        try:
            await retention_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="AI Document Intelligence", version="1.0.0", lifespan=lifespan)
# The original engine writes relative report paths. Serialize runs and set the
# working directory to each job so reports cannot overwrite another job.
_engine_lock = threading.Lock()


@app.middleware("http")
async def protect_api_responses(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def now():
    return datetime.now(timezone.utc).isoformat()

def read_job(job_id: str, owner_id: str | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "Analysis job not found")
    path = JOBS / job_id / "job.json"
    if not path.exists(): raise HTTPException(404, "Analysis job not found")
    job = json.loads(path.read_text(encoding="utf-8"))
    if owner_id is not None and not local_auth_enabled() and job.get("owner_id") != owner_id:
        raise HTTPException(404, "Analysis job not found")
    return job

def write_job(job: dict):
    path = JOBS / job["job_id"] / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def safe_name(name: str) -> str:
    name = Path(name.replace("\\","/")).name
    name = re.sub(r"[^A-Za-z0-9._() -]", "_", name).strip(" .")
    return name[:180] or "upload"

def update(job_id: str, **values):
    job = read_job(job_id); job.update(values); job["updated_at"] = now(); write_job(job); return job

def _document_needs_review(document: dict) -> bool:
    return str(document.get("review_status", "pending")).strip().lower() != "verified"

def _with_document_counts(job: dict) -> dict:
    normalized = dict(job)
    results = normalized.get("results")
    if results is not None:
        normalized_results = []
        for document in results:
            normalized_document = dict(document)
            if not normalized_document.get("review_status"):
                normalized_document["review_status"] = "pending"
            normalized_results.append(normalized_document)
        normalized["results"] = normalized_results
        normalized["review_documents"] = sum(
            1 for document in normalized_results if _document_needs_review(document)
        )
        normalized["failed_documents"] = sum(
            1 for document in normalized_results
            if str(document.get("extraction_status", "")).upper() == "FAILED"
        )
        normalized["successful_documents"] = len(normalized_results) - normalized["failed_documents"]
        if normalized.get("status") in ("completed", "pending_review"):
            if normalized["review_documents"]:
                normalized["status"] = "pending_review"
                normalized["stage"] = (
                    f"Awaiting human verification "
                    f"({normalized['review_documents']} pending)"
                )
            else:
                normalized["status"] = "completed"
    else:
        normalized.setdefault("review_documents", normalized.get("failed_documents", 0))
    return normalized

def summary_model_status() -> dict:
    provider = os.getenv("SUMMARY_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        api_key_configured = bool(api_key and not api_key.lower().startswith("replace_"))
        consent = os.getenv("GEMINI_UNPAID_API_ACKNOWLEDGED", "").strip().lower() in {
            "1", "true", "yes"
        }
        return {
            "provider": "Google Gemini API",
            "model": model,
            "api_key_configured": api_key_configured,
            "unpaid_api_acknowledged": consent,
            "summary_api_ready": api_key_configured and consent,
        }
    return {
        "provider": "Local Ollama" if provider == "ollama" else provider,
        "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        "api_key_configured": True,
        "unpaid_api_acknowledged": False,
        "summary_api_ready": provider == "ollama",
    }

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "local_mode": local_auth_enabled(),
        **summary_model_status(),
    }

@app.get("/api/public-config")
def public_config():
    return {
        "supabase_url": os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
        "local_mode": local_auth_enabled(),
    }

@app.get("/api/dashboard")
def dashboard(user: dict[str, str] = Depends(current_user)):
    jobs = []
    for p in JOBS.iterdir():
        f=p/"job.json"
        if f.exists():
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
                if local_auth_enabled() or job.get("owner_id") == user["id"]:
                    jobs.append(_with_document_counts(job))
            except (OSError, json.JSONDecodeError, TypeError):
                log.exception("Invalid job metadata: %s", p)
    return {
        "total_documents":sum(len(j.get("files",[])) for j in jobs),
        "successful_documents":sum(j.get("successful_documents",0) for j in jobs),
        "failed_documents":sum(j.get("failed_documents",0) for j in jobs),
        "review_documents":sum(j.get("review_documents",j.get("failed_documents",0)) for j in jobs),
        "total_reports":sum(len(j.get("reports",[])) for j in jobs),
        **summary_model_status(),
        "active_jobs":sum(j.get("status") in ("queued","processing") for j in jobs),
        "recent_jobs":sorted(jobs,key=lambda x:x.get("created_at",""),reverse=True)[:8]
    }

@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    user: dict[str, str] = Depends(current_user),
):
    if not files: raise HTTPException(400,"Choose at least one file.")
    if len(files)>MAX_FILES: raise HTTPException(413,f"Maximum {MAX_FILES} files per job.")
    job_id=uuid.uuid4().hex
    job_dir=JOBS/job_id; uploads=job_dir/"uploads"; uploads.mkdir(parents=True)
    saved=[]
    job_total = 0
    upload_quota_reserved = False
    try:
        for index, item in enumerate(files,1):
            name=safe_name(item.filename or f"document_{index}")
            ext=Path(name).suffix.lower()
            if ext not in ALLOWED: raise HTTPException(415,f"Unsupported file type: {name}")
            file_folder=uploads/f"{index:02d}"
            file_folder.mkdir(parents=True, exist_ok=True)
            dest=file_folder/name
            total=0
            with dest.open("wb") as out:
                while True:
                    chunk=await item.read(1024*1024)
                    if not chunk: break
                    total+=len(chunk)
                    job_total += len(chunk)
                    if total>MAX_FILE_BYTES: raise HTTPException(413,f"{name} exceeds upload limit.")
                    if job_total>MAX_TOTAL_UPLOAD_BYTES:
                        raise HTTPException(
                            413,
                            f"Combined upload exceeds {MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)} MB.",
                        )
                    out.write(chunk)
            if total==0: raise HTTPException(400,f"{name} is empty.")
            saved.append({"file_id":uuid.uuid4().hex,"filename":name,"stored_name":str(dest.relative_to(uploads)),"size":total,"extension":ext,"selected":True})
        if not local_auth_enabled():
            consume_upload_quota(user["id"])
            upload_quota_reserved = True
        job={"job_id":job_id,"owner_id":user["id"],"created_at":now(),"updated_at":now(),"status":"uploaded","stage":"Ready to analyze","progress":0,"files":saved,"reports":[],"results":None,"final_summary":"","successful_documents":0,"failed_documents":0,"review_documents":0,"error":None}
        write_job(job)
        return job
    except Exception:
        if upload_quota_reserved:
            _refund_upload_quota(user["id"])
        shutil.rmtree(job_dir,ignore_errors=True); raise

def _parse_sections(text: str) -> list[dict]:
    # Display the exact report text, while deriving section blocks for the UI.
    headings=re.split(r"(?m)^#{1,4}\s+", text or "")
    return [{"text":x.strip()} for x in headings if x.strip()]

def run_analysis(job_id: str):
    job=read_job(job_id)
    update(job_id,status="processing",stage="Starting LangGraph workflow",progress=5,error=None)
    job_dir=JOBS/job_id
    upload_paths=[str(job_dir/"uploads"/x["stored_name"]) for x in job["files"]]
    old_cwd=os.getcwd()
    try:
        # Import lazily: the API health/dashboard can start even if optional OCR
        # dependencies are unavailable; analysis reports the import failure clearly.
        import sys
        sys.path.insert(0,str((BASE/"backend").resolve()))
        import analyzer_engine as engine
        engine.validate_summary_configuration()
        with _engine_lock:
            os.chdir(job_dir)
            update(job_id,stage="Extracting documents and OCR where required",progress=15)
            graph=engine.build_agent_graph()
            initial={"input_files":upload_paths,"selected_files":upload_paths,"files_data":[],"individual_analyses":[],"verified_analyses":[],"final_summary":"","output_format":"pdf","output_paths":[],"archive_name":"","archive_files":[]}
            update(job_id,stage="Analyzing and verifying with Gemini API / LangGraph",progress=25)
            result=graph.invoke(initial)
            update(job_id,stage="Building executive summary and individual PDF reports",progress=90)
            paths=[]
            for raw in result.get("output_paths",[]):
                p=Path(raw)
                if not p.is_absolute(): p=(job_dir/p).resolve()
                if p.exists() and p.suffix.lower()==".pdf": paths.append(p)
            verified=result.get("verified_analyses") or result.get("individual_analyses") or []
            files_data=result.get("files_data") or []
            source_map={x.get("filename"):x for x in files_data}
            report_by_name={}
            for p in paths:
                # Original PDF names are generated in order; associate by index.
                idx=len(report_by_name)
                if idx<len(verified): report_by_name[verified[idx].get("filename","")]=p
            docs=[]
            for item in verified:
                fname=item.get("filename","Unknown")
                source=source_map.get(fname,{})
                analysis=item.get("analysis","")
                pdf=report_by_name.get(fname)
                docs.append({"filename":fname,"type":item.get("type","unknown"),"analysis":analysis,"extraction_status":item.get("extraction_status",source.get("extraction_status","UNKNOWN")),"report_id":f"{job_id[:8]}-{pdf.stem}" if pdf else None,"report_filename":pdf.name if pdf else None,"review_status":"pending","reviewed_at":None})
            reports=[]
            for p in paths:
                original=next((d["filename"] for d in docs if d.get("report_filename")==p.name),"")
                reports.append({"report_id":f"{job_id[:8]}-{p.stem}","filename":p.name,"original_filename":original,"size":p.stat().st_size})
            failed=sum(1 for d in docs if str(d.get("extraction_status","")).upper()=="FAILED")
            review=sum(1 for d in docs if _document_needs_review(d))
            status="pending_review" if review else "completed"
            stage=f"Awaiting human verification ({review} pending)" if review else "Completed"
            update(job_id,status=status,stage=stage,progress=100,results=docs,final_summary=result.get("final_summary",""),reports=reports,successful_documents=len(docs)-failed,failed_documents=failed,review_documents=review,error=None)
    except Exception as exc:
        log.exception("Analysis job %s failed",job_id)
        update(job_id,status="failed",stage="Analysis failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        try: os.chdir(old_cwd)
        except Exception: pass

@app.post("/api/jobs/{job_id}/analyze")
def analyze(
    job_id: str,
    background: BackgroundTasks,
    user: dict[str, str] = Depends(current_user),
):
    job=read_job(job_id, user["id"])
    if job["status"] in ("queued","processing"): raise HTTPException(409,"This job is already running.")
    if _with_document_counts(job)["status"]=="pending_review": raise HTTPException(409,"Verify the current results before restarting this analysis.")
    quota_reserved = not local_auth_enabled()
    if quota_reserved:
        consume_daily_quota(user["id"])
    try:
        update(job_id,status="queued",stage="Queued for analysis",progress=1)
        background.add_task(run_analysis,job_id)
    except Exception:
        if quota_reserved:
            _refund_daily_quota(user["id"])
        raise
    return {"job_id":job_id,"status":"queued"}

@app.get("/api/jobs/{job_id}/status")
def status(job_id: str, user: dict[str, str] = Depends(current_user)):
    j=_with_document_counts(read_job(job_id, user["id"]))
    return {k:j.get(k) for k in ("job_id","status","stage","progress","error","updated_at","successful_documents","failed_documents","review_documents")}

@app.get("/api/jobs/{job_id}/results")
def results(job_id: str, user: dict[str, str] = Depends(current_user)):
    j=read_job(job_id, user["id"])
    if _with_document_counts(j).get("status") not in ("completed","pending_review","failed"): return JSONResponse({"status":j["status"],"stage":j.get("stage")},status_code=202)
    return _with_document_counts(j)

@app.post("/api/jobs/{job_id}/documents/{document_index}/verify")
def verify_document(
    job_id: str,
    document_index: int,
    user: dict[str, str] = Depends(current_user),
):
    job = read_job(job_id, user["id"])
    if job.get("status") not in ("completed", "pending_review"):
        raise HTTPException(409, "The analysis must finish before documents can be verified.")
    results = job.get("results")
    if not isinstance(results, list) or not 0 <= document_index < len(results):
        raise HTTPException(404, "Analyzed document not found")
    updated_results = [dict(document) for document in results]
    document = updated_results[document_index]
    if str(document.get("review_status", "pending")).lower() == "verified":
        return _with_document_counts(job)
    document["review_status"] = "verified"
    document["reviewed_at"] = now()
    remaining = sum(1 for item in updated_results if _document_needs_review(item))
    failed = sum(
        1 for item in updated_results
        if str(item.get("extraction_status", "")).upper() == "FAILED"
    )
    updated = update(
        job_id,
        results=updated_results,
        status="pending_review" if remaining else "completed",
        stage=f"Awaiting human verification ({remaining} pending)" if remaining else "Human verification complete",
        progress=100,
        successful_documents=len(updated_results) - failed,
        failed_documents=failed,
        review_documents=remaining,
    )
    return _with_document_counts(updated)

@app.get("/api/jobs/{job_id}/reports")
def reports(job_id: str, user: dict[str, str] = Depends(current_user)):
    return read_job(job_id, user["id"]).get("reports",[])

def locate_report(report_id: str, owner_id: str):
    for folder in JOBS.iterdir():
        jobfile=folder/"job.json"
        if not jobfile.exists(): continue
        try:
            job=json.loads(jobfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("Invalid job metadata: %s", jobfile)
            continue
        if not local_auth_enabled() and job.get("owner_id") != owner_id:
            continue
        for report in job.get("reports",[]):
            if report.get("report_id")==report_id:
                path=folder/"output"/"individual_reports"/report["filename"]
                if not path.exists():
                    # The engine output path is relative to the job CWD.
                    path=folder/"output"/"individual_reports"/report["filename"]
                return path,report
    raise HTTPException(404,"PDF report not found")

@app.get("/api/reports/{report_id}/view")
def view_report(report_id: str, user: dict[str, str] = Depends(current_user)):
    path,meta=locate_report(report_id, user["id"])
    return FileResponse(path,media_type="application/pdf",headers={"Content-Disposition":f'inline; filename="{meta["filename"]}"'})

@app.get("/api/reports/{report_id}/download")
def download_report(report_id: str, user: dict[str, str] = Depends(current_user)):
    path,meta=locate_report(report_id, user["id"])
    return FileResponse(path,media_type="application/pdf",filename=meta["filename"])

@app.get("/api/jobs/{job_id}/download-all")
def download_all(job_id: str, user: dict[str, str] = Depends(current_user)):
    j=read_job(job_id, user["id"]); dest=JOBS/job_id/"all_reports.zip"
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        for r in j.get("reports",[]):
            p=JOBS/job_id/"output"/"individual_reports"/r["filename"]
            if p.exists(): z.write(p,r["filename"])
    return FileResponse(dest,media_type="application/zip",filename=f"analysis_{job_id[:8]}_reports.zip")

@app.get("/api/history")
def history(user: dict[str, str] = Depends(current_user)):
    items=[]
    for p in JOBS.iterdir():
        f=p/"job.json"
        if f.exists():
            try:
                job=json.loads(f.read_text(encoding="utf-8"))
                if local_auth_enabled() or job.get("owner_id") == user["id"]:
                    items.append(_with_document_counts(job))
            except (OSError, json.JSONDecodeError):
                log.exception("Invalid job metadata: %s", f)
    return sorted(items,key=lambda x:x.get("created_at",""),reverse=True)

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, user: dict[str, str] = Depends(current_user)):
    job = read_job(job_id, user["id"])
    if job.get("status") in ("queued", "processing"):
        raise HTTPException(409, "Wait for the analysis to finish before deleting this job.")
    try:
        shutil.rmtree(JOBS / job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Analysis job not found") from exc
    except OSError as exc:
        log.exception("Unable to delete analysis job %s", job_id)
        raise HTTPException(500, "Unable to delete the analysis job.") from exc
    return {"deleted": True}

@app.get("/api/settings")
def settings(user: dict[str, str] = Depends(current_user)):
    return {
        **summary_model_status(),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "max_upload_mb": MAX_FILE_BYTES // (1024 * 1024),
        "max_upload_total_mb": MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024),
        "analyses_per_day": MAX_ANALYSES_PER_USER_PER_DAY,
        "upload_jobs_per_day": MAX_ANALYSES_PER_USER_PER_DAY,
        "job_retention_days": JOB_RETENTION_DAYS,
        "allowed_extensions": sorted(ALLOWED),
    }

# Serve built React app if present.
DIST=BASE/"frontend"/"dist"
if DIST.exists():
    app.mount("/",StaticFiles(directory=DIST,html=True),name="frontend")
