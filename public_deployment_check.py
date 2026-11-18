from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric import rsa

from backend import api as web_api
from backend import auth


class PublicDeploymentChecks(unittest.TestCase):
    def test_supabase_hs256_token_requires_valid_issuer_and_audience(self):
        user_id = str(uuid.uuid4())
        issuer = "https://example.supabase.co/auth/v1"
        token = jwt.encode(
            {
                "sub": user_id,
                "iss": issuer,
                "aud": "authenticated",
                "exp": int(time.time()) + 60,
            },
            "test-secret-with-at-least-32-bytes",
            algorithm="HS256",
        )
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_JWT_SECRET": "test-secret-with-at-least-32-bytes",
                "AUTH_REQUIRED": "true",
                "HOST": "0.0.0.0",
            },
        ):
            self.assertEqual(auth.current_user(credentials), {"id": user_id})
            invalid = jwt.encode(
                {
                    "sub": user_id,
                    "iss": issuer,
                    "aud": "wrong-audience",
                    "exp": int(time.time()) + 60,
                },
                "test-secret-with-at-least-32-bytes",
                algorithm="HS256",
            )
            with self.assertRaises(HTTPException) as raised:
                auth.current_user(
                    HTTPAuthorizationCredentials(scheme="Bearer", credentials=invalid)
                )
            self.assertEqual(raised.exception.status_code, 401)

    def test_supabase_rs256_token_is_verified_using_the_jwks_key(self):
        user_id = str(uuid.uuid4())
        issuer = "https://example.supabase.co/auth/v1"
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        token = jwt.encode(
            {
                "sub": user_id,
                "iss": issuer,
                "aud": "authenticated",
                "exp": int(time.time()) + 60,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )
        jwks = Mock()
        jwks.get_signing_key_from_jwt.return_value = SimpleNamespace(key=public_key)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://example.supabase.co", "AUTH_REQUIRED": "true"},
        ), patch.object(auth, "_jwks_client", return_value=jwks):
            self.assertEqual(auth.current_user(credentials), {"id": user_id})

    def test_private_api_routes_require_authentication(self):
        public_paths = {"/api/health", "/api/public-config"}
        protected = [
            route
            for route in web_api.app.routes
            if getattr(route, "path", "").startswith("/api/")
            and route.path not in public_paths
        ]
        self.assertTrue(protected)
        for route in protected:
            dependencies = {item.call for item in route.dependant.dependencies}
            self.assertIn(auth.current_user, dependencies, route.path)

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_api, "JOBS", Path(temp_dir) / "jobs"
        ):
            web_api.JOBS.mkdir()
            with patch.dict(os.environ, {"AUTH_REQUIRED": "true", "HOST": "0.0.0.0"}):
                with TestClient(web_api.app) as client:
                    self.assertEqual(client.get("/api/dashboard").status_code, 401)

    def test_jobs_and_reports_are_scoped_to_the_owner(self):
        owner_id = str(uuid.uuid4())
        other_user_id = str(uuid.uuid4())
        job_id = "a" * 32
        report_id = "private-report"
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_api, "JOBS", Path(temp_dir) / "jobs"
        ), patch.object(web_api, "local_auth_enabled", return_value=False) as local_mode:
            folder = web_api.JOBS / job_id
            report_folder = folder / "output" / "individual_reports"
            report_folder.mkdir(parents=True)
            report_path = report_folder / "report.pdf"
            report_path.write_bytes(b"%PDF-test")
            (folder / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "owner_id": owner_id,
                        "reports": [
                            {"report_id": report_id, "filename": report_path.name}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(web_api.read_job(job_id, owner_id)["owner_id"], owner_id)
            with self.assertRaises(HTTPException) as raised:
                web_api.read_job(job_id, other_user_id)
            self.assertEqual(raised.exception.status_code, 404)
            other_job_id = "d" * 32
            other_folder = web_api.JOBS / other_job_id
            other_folder.mkdir()
            (other_folder / "job.json").write_text(
                json.dumps({"job_id": other_job_id, "owner_id": other_user_id}),
                encoding="utf-8",
            )
            self.assertEqual(
                [job["job_id"] for job in web_api.dashboard({"id": owner_id})["recent_jobs"]],
                [job_id],
            )
            self.assertEqual(
                web_api.locate_report(report_id, owner_id)[0],
                report_path,
            )
            with self.assertRaises(HTTPException) as raised:
                web_api.locate_report(report_id, other_user_id)
            self.assertEqual(raised.exception.status_code, 404)
            legacy_job_id = "e" * 32
            legacy_folder = web_api.JOBS / legacy_job_id
            legacy_folder.mkdir()
            (legacy_folder / "job.json").write_text(
                json.dumps({"job_id": legacy_job_id}),
                encoding="utf-8",
            )
            local_mode.return_value = True
            self.assertEqual(
                web_api.read_job(legacy_job_id, "local-user")["job_id"],
                legacy_job_id,
            )

    def test_daily_analysis_quota_is_atomic_and_per_user(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_api, "_USAGE_DB", Path(temp_dir) / "usage.sqlite3"
        ), patch.object(web_api, "MAX_ANALYSES_PER_USER_PER_DAY", 3):
            user_id = str(uuid.uuid4())
            self.assertEqual(web_api.consume_daily_quota(user_id), 1)
            self.assertEqual(web_api.consume_daily_quota(user_id), 2)
            self.assertEqual(web_api.consume_daily_quota(user_id), 3)
            with self.assertRaises(HTTPException) as raised:
                web_api.consume_daily_quota(user_id)
            self.assertEqual(raised.exception.status_code, 429)
            self.assertEqual(web_api.consume_upload_quota(user_id), 1)
            self.assertEqual(web_api.consume_upload_quota(user_id), 2)
            self.assertEqual(web_api.consume_upload_quota(user_id), 3)
            with self.assertRaises(HTTPException) as raised:
                web_api.consume_upload_quota(user_id)
            self.assertEqual(raised.exception.status_code, 429)
            self.assertEqual(web_api.consume_daily_quota(str(uuid.uuid4())), 1)

    def test_retention_removes_expired_jobs_but_keeps_active_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_api, "JOBS", Path(temp_dir) / "jobs"
        ), patch.object(web_api, "JOB_RETENTION_DAYS", 7), patch.object(
            web_api, "local_auth_enabled", return_value=False
        ):
            web_api.JOBS.mkdir()
            expired_id = "b" * 32
            active_id = "c" * 32
            old_timestamp = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
            for job_id, status in ((expired_id, "completed"), (active_id, "processing")):
                folder = web_api.JOBS / job_id
                folder.mkdir()
                (folder / "job.json").write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "created_at": old_timestamp,
                            "status": status,
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertEqual(web_api.cleanup_expired_jobs(), 1)
            self.assertFalse((web_api.JOBS / expired_id).exists())
            self.assertTrue((web_api.JOBS / active_id).exists())
            local_only_id = "f" * 32
            local_folder = web_api.JOBS / local_only_id
            local_folder.mkdir()
            (local_folder / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": local_only_id,
                        "created_at": old_timestamp,
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(web_api, "local_auth_enabled", return_value=True):
                self.assertEqual(web_api.cleanup_expired_jobs(), 0)
            self.assertTrue(local_folder.exists())


if __name__ == "__main__":
    unittest.main()
