# DocuMind — AI Document Intelligence

A local web interface for a LangGraph multi-file analyzer. Text reports use Google's Gemini Developer API by default; local Ollama remains available as an alternative and for optional vision fallback. The API invokes `backend/analyzer_engine.py` and generates individual PDF reports.

## Requirements
- Windows 10/11 (or Linux/macOS), Python 3.11 recommended
- A Google AI Studio API key for the default Gemini summary provider
- Or Ollama installed and running if `SUMMARY_PROVIDER=ollama`
- PyMuPDF (included in `requirements.txt`) to render scanned PDF pages
- Tesseract OCR installed on Windows, plus `pytesseract` from `requirements.txt`, is recommended for scanned PDFs. The standard `C:\Program Files\Tesseract-OCR` installation is auto-detected.
- Ollama vision fallback (default `llava-phi3`) is used only when PaddleOCR and Tesseract are unavailable; those results are marked `REVIEW_REQUIRED`

## Start on Windows
1. Open PowerShell in this folder.
2. Run `Set-ExecutionPolicy -Scope Process Bypass` if script execution is blocked.
3. Run `./start.ps1`.
4. For a non-standard Tesseract installation, set `TESSERACT_CMD` to the full executable path.

Or manually:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```
`python main.py` starts FastAPI at `http://127.0.0.1:8000` and opens your browser. Keep the terminal open while using the application.

## Share DocuMind on a trusted local network
For same-Wi-Fi access, configure Supabase sign-in first. The server refuses to disable authentication when `HOST` is not loopback. Set `AUTH_REQUIRED=true`, `HOST=0.0.0.0`, and `PORT=8000`, provide `SUPABASE_URL` and `SUPABASE_ANON_KEY`, then add a Windows Firewall rule for Private networks only. Open the host's private IPv4 address from the other laptop. Never expose port 8000 through a router.

## Deploy for access from anywhere (Railway + Supabase)
The app includes Supabase email sign-in, verified accounts, per-account job/report isolation, up to three upload jobs and three analysis runs per account per UTC day, and automatic deletion of jobs seven days after upload. The Docker image builds the React frontend and installs Tesseract. Use a Railway persistent volume mounted at `/app/data`; without it, uploads and reports are not durable. Keep one service replica: job data and quota accounting use the attached filesystem and SQLite; horizontal scaling requires shared storage, a shared database, and a distributed job queue.

1. **Create a GitHub repository.** Push the project source, but never commit `.env`, API keys, `data`, `.venv`, or `node_modules`. The included `.gitignore` and `.dockerignore` exclude these paths.
2. **Create a Supabase project.** Enable email/password sign-up and email confirmations. Set the Supabase Auth **Site URL** and allowed redirect URL to your eventual Railway HTTPS domain. Configure a production SMTP provider in Supabase for dependable verification and password-reset email delivery; the built-in test mailer has sending restrictions. Copy the Project URL and public anon/publishable key. Do not use the Supabase service-role key in the browser.
3. **Create a Railway project from the GitHub repository.** Railway detects the included `Dockerfile`. Generate a public HTTPS domain in the service's Networking settings.
4. **Attach a Railway volume** to the app service, mounted at `/app/data`. Set `DOCUMIND_DATA_DIR=/app/data`. This volume stores per-user uploads, reports, job history, and the daily quota database.
5. **Add environment variables in Railway** (never in source control):
   - `GEMINI_API_KEY` — your server-side Gemini key; all users share your quota and usage costs.
   - `GEMINI_UNPAID_API_ACKNOWLEDGED=true` only after you read the applicable Gemini API terms.
   - `SUPABASE_URL` and `SUPABASE_ANON_KEY` — from the Supabase project.
   - `SUPABASE_JWT_SECRET` only if that Supabase project uses legacy HS256 JWT signing; asymmetric signing keys are verified through Supabase's JWKS endpoint.
   - `AUTH_REQUIRED=true`, `HOST=0.0.0.0`, `DOCUMIND_DATA_DIR=/app/data`.
   - `MAX_ANALYSES_PER_USER_PER_DAY=3`, `JOB_RETENTION_DAYS=7`, `MAX_UPLOAD_MB=25`, `MAX_UPLOAD_FILES=10`, and `MAX_UPLOAD_TOTAL_MB=100`.
   - Optional model settings such as `GEMINI_MODEL` and `GEMINI_API_TIMEOUT`.
6. After Railway creates the HTTPS domain, add it to Supabase's allowed redirect URLs and Site URL, then redeploy. Open the Railway URL, create an account, verify its email, and test that a second account cannot see the first account's jobs.

The API rejects unauthenticated access to dashboards, uploads, jobs, reports, and settings. The public health and client-configuration endpoints do not expose server secrets. Extracted document text is still sent to Gemini using your key; do not make a public service available for sensitive documents unless your Gemini plan and privacy terms are suitable.

**Cost and free-tier note:** Railway currently lists a $5/month Hobby subscription with $5 of included resource usage; volume storage and usage above the included amount can cost extra, and Gemini API charges are separate. Check [Railway pricing](https://docs.railway.com/pricing/plans) before deploying. Render's Free web services sleep after 15 minutes and lose local files when they sleep or restart, and do not support persistent disks; that is suitable only for a temporary demo, not this app's retained private reports. See [Render free-service limits](https://render.com/docs/free) and [Railway volumes](https://docs.railway.com/volumes).

## Frontend development
```powershell
cd frontend
npm install
npm run dev
```
The Vite dev server currently serves the frontend on port 5173; for local API integration, use the built frontend served by FastAPI, or configure a Vite proxy for `/api` to `http://127.0.0.1:8000`.

Build production frontend:
```powershell
cd frontend
npm install
npm run build
cd ..
python main.py
```
FastAPI serves `frontend/dist` automatically when it exists.

## Workflow
1. Upload files in the browser (multiple files and ZIP supported).
2. Start analysis. The API stores the upload in `data/jobs/<job_id>/uploads`.
3. The existing LangGraph pipeline extracts, analyzes, verifies, synthesizes and generates draft individual PDFs.
4. The job enters `Needs review`; compare each analysis and PDF with its source, then mark each document as verified.
5. Once all documents are verified, the job advances to `Completed`. Draft PDFs can be previewed or downloaded while review is pending.
6. Hosted jobs are retained for seven days and then automatically deleted; local jobs remain until you delete them. Existing completed reports without a saved human-review decision are treated as pending review.
7. Open an `Uploaded` or `Failed` job from history to start or retry its analysis. Pending-review jobs must be verified before analysis can be restarted; queued or processing jobs must finish first.

By default, Gemini generates source-grounded study reports with a title, introduction, concepts, examples, analysis or tables when relevant, findings, revision points, conclusion, and key takeaway. Inapplicable sections are omitted. Long files are split into source segments and combined without a second model call. Scanned PDFs send only high-confidence OCR pages to Gemini; low-confidence pages are excluded and marked `REVIEW_REQUIRED`. Image OCR filters out very low-confidence words, keeps extraction diagnostics out of the model's source text, and notes when important details should be checked against the original. If all OCR methods fail, the report shows a reader-friendly limitation instead of internal OCR engine errors. The collection overview is assembled from the individual reports without another API request. Accuracy is not certified as a percentage; verify source references and OCR warnings against the originals.

## Gemini free-tier privacy and setup
1. Create an API key in [Google AI Studio](https://aistudio.google.com/apikey). Use a project with billing disabled if you need to stay within the free quota; quotas and model access can change, and Google may charge requests made through a billable project.
2. Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and set `GEMINI_UNPAID_API_ACKNOWLEDGED=true` only after reading Google's [Gemini API terms](https://aistudio.google.com/docs/terms).
3. Restart the application after changing `.env`.

**Important:** Google states that inputs and outputs on unpaid Gemini services may be used to improve its products and may be reviewed by humans. Do not send sensitive, confidential, or personal information to the free API. This application sends extracted document text, not the original PDF binary. If those terms are unsuitable, use `SUMMARY_PROVIDER=ollama` to keep text summaries local.

## Configuration
Copy `.env.example` to `.env` if needed. The API loads the project-root `.env` at startup; restart the app after changing it. Common settings:
- `SUMMARY_PROVIDER=gemini` (default) or `ollama`
- `GEMINI_MODEL=gemini-3.5-flash-lite`
- `GEMINI_API_KEY=...`
- `GEMINI_UNPAID_API_ACKNOWLEDGED=false` (must be explicitly changed to `true` for free-tier requests)
- `GEMINI_CHUNK_CHARS=16000`
- `OLLAMA_MODEL=qwen3:4b` (local summary option)
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_VISION_MODEL=llava-phi3`
- `HOST=127.0.0.1` (use `0.0.0.0` only when intentionally sharing on a trusted network)
- `PORT=8000`
- `AUTH_REQUIRED=false` is allowed only on loopback; Docker sets it to `true`
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and optionally `SUPABASE_JWT_SECRET` for hosted sign-in
- `MAX_ANALYSES_PER_USER_PER_DAY=3`
- `JOB_RETENTION_DAYS=7`
- `DOCUMIND_DATA_DIR=./data`
- `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` (only needed for non-standard installations)
- `MAX_UPLOAD_MB=25`
- `MAX_UPLOAD_FILES=10`
- `MAX_UPLOAD_TOTAL_MB=100`

## Important implementation notes
- Job executions are serialized because the existing PDF generator writes to a relative `output/individual_reports` path. The backend changes into a unique job directory while invoking the graph, preventing cross-job report collisions.
- If the server restarts while a background analysis is queued or processing, the job is marked failed and can be retried; the API does not claim that an in-memory task survived a restart.
- The LangGraph extraction, verification and report-generation nodes remain the workflow backbone. Text summarization uses Gemini or local Ollama according to `SUMMARY_PROVIDER`; the web workflow does not invoke the console file picker or output-format prompt.
- The processing stage and completion status are real backend state. The UI does not invent intermediate percentage updates; progress advances at the backend's explicit workflow boundaries.
- ZIP entries are checked for traversal paths, entry count, expansion size, and suspicious compression ratios. Public deployments still need routine dependency updates and monitoring.
- In local loopback mode, authentication is disabled for the single local user. A non-loopback bind requires `AUTH_REQUIRED=true`; hosted deployments use Supabase JWT verification and per-user job ownership.
- Hosted jobs, source uploads, and PDF reports are automatically deleted seven days after creation. Users can also delete individual jobs earlier.
- The sample scanned PDF was reprocessed locally with Tesseract and the updated report pipeline.
