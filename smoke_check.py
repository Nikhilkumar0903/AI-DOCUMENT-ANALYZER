from pathlib import Path
import sys
from unittest.mock import Mock, patch
from fastapi import HTTPException
root=Path(__file__).parent
assert (root/'backend/analyzer_engine.py').exists()
s=(root/'backend/analyzer_engine.py').read_text(encoding='utf-8')
for token in ('def build_agent_graph(', 'def ingest_files_node(', 'def analyze_files_node(', 'def verify_analyses_node(', 'def synthesize_report_node(', 'def generate_pdf_node('):
    assert token in s, f'Missing original pipeline function: {token}'
api=(root/'backend/api.py').read_text(encoding='utf-8')
for token in ('graph.invoke(initial)', '"output_format":"pdf"', '@app.post("/api/upload")', '@app.get("/api/jobs/{job_id}/results")', 'summary_model_status'):
    assert token in api, f'Missing web integration contract: {token}'
app_source=(root/'frontend/src/App.tsx').read_text(encoding='utf-8')
assert "api(`/jobs/${id}`,{method:'DELETE'})" in app_source
assert 'disabled={!selected.length||busy}' in app_source
assert 'disabled={!selected.length||busy||!health?.summary_api_ready}' not in app_source
sys.path.insert(0, str(root/'backend'))
import analyzer_engine
from backend import api as web_api

delete_route=next(route for route in web_api.app.routes if getattr(route, 'path', None)=='/api/jobs/{job_id}')
assert delete_route.methods == {'DELETE'}
with patch.object(web_api, 'read_job', return_value={'status':'processing'}), \
     patch.object(web_api.shutil, 'rmtree') as remove_tree:
    try:
        web_api.delete_job('a' * 32, {"id": "test-user"})
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError('An active analysis must not be deleted.')
    remove_tree.assert_not_called()

legacy_review = web_api._with_document_counts({
    "job_id": "a" * 32,
    "status": "completed",
    "results": [
        {"filename": "one.txt", "analysis": "A generated report.", "extraction_status": "SUCCESS"},
        {"filename": "two.txt", "analysis": "Another generated report.", "extraction_status": "SUCCESS"},
    ],
})
assert legacy_review["status"] == "pending_review"
assert legacy_review["review_documents"] == 2
assert legacy_review["successful_documents"] == 2

verification_job = {
    "job_id": "b" * 32,
    "status": "pending_review",
    "results": [
        {
            "filename": "one.txt",
            "analysis": "A generated report.",
            "extraction_status": "SUCCESS",
            "review_status": "pending",
        },
        {
            "filename": "two.txt",
            "analysis": "Another generated report.",
            "extraction_status": "SUCCESS",
            "review_status": "pending",
        },
    ],
}
def update_verification_job(_job_id, **values):
    verification_job.update(values)
    return verification_job

with patch.object(web_api, "read_job", return_value=verification_job), \
     patch.object(web_api, "update", side_effect=update_verification_job):
    test_user = {"id": "test-user"}
    partially_verified = web_api.verify_document(verification_job["job_id"], 0, test_user)
    assert partially_verified["status"] == "pending_review"
    assert partially_verified["review_documents"] == 1
    verified_job = web_api.verify_document(verification_job["job_id"], 1, test_user)
assert verified_job["status"] == "completed"
assert verified_job["review_documents"] == 0
assert verified_job["successful_documents"] == 2
assert all(item["review_status"] == "verified" for item in verified_job["results"])

summary=analyzer_engine.synthesize_report_node({
    'verified_analyses': [{
        'filename': 'sample.txt',
        'type': 'Text',
        'extraction_status': 'SUCCESS',
        'analysis': '### Plain-Language Summary\nThe document lists annual expenses and projected costs.'
    }]
})['final_summary']
assert 'annual expenses and projected costs' in summary
assert 'mining-area reports' not in summary

structured_summary = analyzer_engine.synthesize_report_node({
    "verified_analyses": [{
        "filename": "structured.txt",
        "type": "Text",
        "extraction_status": "SUCCESS",
        "analysis": (
            "### 1. Document Title\nStudy Notes\n\n"
            "### 2. Introduction\nThe source explains basic information theory.\n\n"
            "### 3. Main Concepts\nDetails follow."
        ),
    }]
})["final_summary"]
assert "The source explains basic information theory." in structured_summary
assert "Details follow." not in structured_summary.split(
    "## Master Executive Summary\n", 1
)[1].split("\n## File-by-File Overview\n", 1)[0]

assert analyzer_engine.analysis_needs_review(
    "### Extraction Limitations\nVision transcription needs verification.",
    "REVIEW_REQUIRED",
)
assert analyzer_engine.analysis_needs_review(
    "We are consolidating summaries of different chunks of ONE source file.",
    "SUCCESS",
)
assert not analyzer_engine.analysis_needs_review(
    "### Reader-Friendly Summary\nThe document lists annual expenses.",
    "SUCCESS",
)
review_summary = analyzer_engine.synthesize_report_node({
    "verified_analyses": [{
        "filename": "file.pdf",
        "type": "PDF",
        "extraction_status": "REVIEW_REQUIRED",
        "analysis": "The vision transcription reported 72 repeated items.",
    }]
})["final_summary"]
assert "Review required: extraction or analysis quality is uncertain." in review_summary
master_summary = review_summary.split("## Master Executive Summary\n", 1)[1]
master_summary = master_summary.split("\n## File-by-File Overview\n", 1)[0]
assert "72 repeated items" not in master_summary

with patch.object(
    analyzer_engine, "extract_pdf_text",
    return_value="--- TEXT TRANSCRIPTION (OLLAMA VISION FALLBACK; VERIFY NUMBERS) ---\ntext",
):
    extracted = analyzer_engine.parse_file("scanned.pdf", b"pdf")
assert extracted["extraction_status"] == "REVIEW_REQUIRED"

gemini_response = Mock(status_code=200)
gemini_response.json.return_value = {
    "modelVersion": "gemini-3.5-flash-lite",
    "candidates": [{
        "finishReason": "STOP",
        "content": {"parts": [{"text": "### Reader-Friendly Summary\nA source-grounded summary."}]},
    }],
}
with patch.object(analyzer_engine, "SUMMARY_PROVIDER", "gemini"), \
     patch.object(analyzer_engine, "GEMINI_API_KEY", "test-api-key"), \
     patch.object(analyzer_engine, "GEMINI_UNPAID_API_ACKNOWLEDGED", True), \
     patch.object(analyzer_engine.requests, "post", return_value=gemini_response) as post:
    gemini_answer = analyzer_engine._invoke_llm_with_retry(
        [analyzer_engine.HumanMessage(content="Summarize only this source.")],
        max_predict_tokens=500,
    )
assert "source-grounded summary" in gemini_answer.content
assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "test-api-key"
assert post.call_args.kwargs["json"]["generationConfig"]["maxOutputTokens"] == 500
assert "test-api-key" not in post.call_args.args[0]

with patch.object(analyzer_engine, "SUMMARY_PROVIDER", "gemini"), \
     patch.object(
         analyzer_engine, "_invoke_llm_with_retry",
         return_value=Mock(
             content="### 3. Main Concepts\nInformation ($n$.) means resolving uncertainty."
         ),
     ) as invoke:
    chunk_summary = analyzer_engine._summarize_source_chunk(
        "sample.pdf", "PDF", "", "Short source text.", 1, 1
    )
assert invoke.call_args.kwargs["max_predict_tokens"] >= 1800
assert "Information, n." in chunk_summary

with patch.object(analyzer_engine, "SUMMARY_PROVIDER", "gemini"), \
     patch.object(analyzer_engine, "GEMINI_API_KEY", "test-api-key"), \
     patch.object(analyzer_engine, "GEMINI_UNPAID_API_ACKNOWLEDGED", False), \
     patch.object(analyzer_engine.requests, "post") as post:
    try:
        analyzer_engine._invoke_llm_with_retry(
            [analyzer_engine.HumanMessage(content="Do not send without consent.")]
        )
    except RuntimeError as exc:
        assert "GEMINI_UNPAID_API_ACKNOWLEDGED" in str(exc)
    else:
        raise AssertionError("Gemini must require free-tier data-use opt-in.")
    post.assert_not_called()

with patch.object(analyzer_engine, "SUMMARY_PROVIDER", "gemini"), \
     patch.object(
         analyzer_engine, "_summarize_source_chunk",
         side_effect=RuntimeError("Gemini API returned HTTP 429"),
     ):
    try:
        analyzer_engine.analyze_files_node({
            "files_data": [{
                "filename": "limited.txt",
                "type": "Text",
                "raw_content": "Source content must not become a fake report.",
            }]
        })
    except RuntimeError as exc:
        assert "Gemini summary generation failed" in str(exc)
    else:
        raise AssertionError("Gemini failures must not produce a success-shaped report.")

ocr_content = (
    "--- Page 1 ---\n"
    "--- OCR EXTRACTED TEXT (TESSERACT FALLBACK) ---\n"
    "Sample PDF Document\nRobert Maron\nFebruary 20, 1999\n"
    "--- Page 2 ---\n"
    "Garbled line 72 XX9\n"
    "[OCR quality warning: Tesseract average confidence is 77%.]"
    "\n--- Page 3 ---\n"
    "1.1 How to compile a .tex file to a .pdf file\n"
    "1. put all source files in one directory\n"
    "chapter title and edit the body of the chapter appropriately."
)
with patch.object(
   analyzer_engine, "_summarize_source_chunk",
   return_value="### Reader-Friendly Summary\nGemini summarized the readable OCR pages.",
) as summarize:
    ocr_analysis = analyzer_engine.analyze_files_node({
        "files_data": [{
            "filename": "scanned.pdf",
            "type": "PDF",
            "raw_content": ocr_content,
            "extraction_status": "REVIEW_REQUIRED",
        }]
    })["individual_analyses"][0]["analysis"]
ocr_source = summarize.call_args.args[3]
assert "February 20, 1999" in ocr_source
assert "1.1 How to compile a .tex file to a .pdf file" in ocr_source
assert "Garbled line 72" not in ocr_source
assert "1. put all source files" in ocr_source
assert "Low-confidence OCR pages" in ocr_analysis

ocr_data = {
    "text": ["13", "1", "D.", "the", "suicide", "king", "3"],
    "conf": ["90", "91", "90", "90", "90", "90", "10"],
    "page_num": ["1"] * 7,
    "block_num": ["1"] * 7,
    "par_num": ["1"] * 7,
    "line_num": ["1"] * 7,
    "left": ["0", "10", "20", "30", "40", "50", "60"],
}
with patch("pytesseract.image_to_data", return_value=ocr_data):
    filtered_ocr_text = analyzer_engine._tesseract_ocr_fallback(
        analyzer_engine.Image.new("RGB", (10, 10))
    )
assert "13 1 D. the suicide king" in filtered_ocr_text
assert "Low-confidence OCR tokens were omitted" in filtered_ocr_text
assert not any(token == "3" for token in filtered_ocr_text.split())

image_ocr_content = (
    "--- OCR EXTRACTED TEXT (TESSERACT FALLBACK) ---\n"
    "Definition of Information.\n"
    "{ # of possibilities remaining\n"
    "1 D. The card is the suicide king\n"
    "[OCR quality warning: Low-confidence OCR tokens were omitted.]\n"
    "Extraction note: PaddleOCR failed; Tesseract fallback was used.\n"
    "PaddleOCR error: native runtime unavailable."
)
with patch.object(
    analyzer_engine, "_summarize_source_chunk",
    return_value="### Plain-Language Summary\nThe image explains information using a card example.",
) as summarize:
    image_analysis = analyzer_engine.analyze_files_node({
        "files_data": [{
            "filename": "image.png",
            "type": "Image",
            "raw_content": image_ocr_content,
            "extraction_status": "REVIEW_REQUIRED",
        }]
    })["individual_analyses"][0]["analysis"]
cleaned_image_source = summarize.call_args.args[3]
assert "PaddleOCR failed" not in cleaned_image_source
assert "Low-confidence OCR tokens were omitted" not in cleaned_image_source
assert "The card is the suicide king" in cleaned_image_source
assert "{ # of possibilities remaining" not in cleaned_image_source
assert "# of possibilities remaining" in cleaned_image_source
assert "PaddleOCR failed" not in image_analysis
assert "Low-confidence OCR tokens were omitted" in image_analysis

failed_image_analysis = analyzer_engine.analyze_files_node({
    "files_data": [{
        "filename": "unreadable.png",
        "type": "Image",
        "raw_content": (
            "=== OCR EXTRACTION FAILED ===\n"
            "PaddleOCR error: native runtime unavailable.\n"
            "Tesseract fallback error: executable not found."
        ),
        "extraction_status": "FAILED",
    }]
})["individual_analyses"][0]["analysis"]
assert "PaddleOCR" not in failed_image_analysis
assert "Tesseract fallback error" not in failed_image_analysis
assert "upload a clearer image or scan" in failed_image_analysis

with patch.object(analyzer_engine, "SUMMARY_PROVIDER", "gemini"), \
     patch.object(
         analyzer_engine, "_summarize_source_chunk",
         return_value="### Reader-Friendly Summary\nGemini summarized the workbook.",
     ) as summarize:
    workbook_analysis = analyzer_engine.analyze_files_node({
        "files_data": [{
            "filename": "sample.xlsx",
            "type": "Excel",
            "raw_content": "Workbook: sample.xlsx\n--- Sheet: Costs ---\nA1=Year | B1=Amount\nA2=2025 | B2=1200",
        }]
    })["individual_analyses"][0]["analysis"]
assert "Gemini summarized the workbook" in workbook_analysis
assert "A2=2025 | B2=1200" in summarize.call_args.args[3]

with patch.object(
    analyzer_engine, "_summarize_source_chunk",
    side_effect=["Summary for first part.", "Summary for second part."],
), patch.object(
    analyzer_engine, "_invoke_llm_with_retry",
    side_effect=AssertionError("Chunk outputs must not be sent to a second LLM merge."),
):
    segmented = analyzer_engine.analyze_files_node({
        "files_data": [{
            "filename": "long.txt",
            "type": "Text",
            "raw_content": "x" * 17000,
        }]
    })["individual_analyses"][0]["analysis"]
assert "Source segment 1 of 2" in segmented
assert "Summary for second part." in segmented
assert "We are consolidating summaries" not in segmented
print('Source contract smoke check passed.')
