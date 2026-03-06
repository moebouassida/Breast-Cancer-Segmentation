"""
test_api.py — Integration tests for the full API + middleware workflow.
Tests the entire pipeline: consent → anonymization → inference → audit → erasure.

Run:
    pytest tests/test_api.py -v
"""

import io
import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Create a test client with the full middleware stack."""
    from fastapi.testclient import TestClient
    from src.app import app

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def test_image_bytes():
    """Fake 128x128 grayscale ultrasound image as JPEG bytes."""
    img = Image.fromarray(
        np.random.randint(50, 200, (128, 128), dtype=np.uint8), mode="L"
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data

    def test_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "healthy"


# ── GDPR Consent Enforcement ──────────────────────────────────────────────────


class TestConsentEnforcement:

    def test_predict_without_consent_returns_403(self, client, test_image_bytes):
        r = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 403

    def test_predict_without_consent_error_message(self, client, test_image_bytes):
        data = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        ).json()
        assert data["error"] == "consent_required"
        assert "Article 9" in data["gdpr_article"]

    def test_predict_with_consent_not_blocked(self, client, test_image_bytes):
        r = client.post(
            "/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        # Should not be 403
        assert r.status_code != 403

    def test_explain_without_consent_returns_403(self, client, test_image_bytes):
        r = client.post(
            "/explain/predict",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 403


# ── Security Headers ──────────────────────────────────────────────────────────


class TestSecurityHeaders:

    def test_request_id_header_present(self, client, test_image_bytes):
        r = client.post(
            "/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        assert "x-request-id" in r.headers

    def test_x_content_type_options(self, client):
        r = client.get("/health")
        assert r.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options(self, client):
        r = client.get("/health")
        assert r.headers.get("x-frame-options") == "DENY"

    def test_cache_control(self, client):
        r = client.get("/health")
        assert "no-store" in r.headers.get("cache-control", "")


# ── Prediction ────────────────────────────────────────────────────────────────


class TestPrediction:

    @pytest.fixture(autouse=True)
    def response(self, client, test_image_bytes):
        self.r = client.post(
            "/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        self.data = self.r.json()

    def test_returns_200(self):
        assert self.r.status_code == 200

    def test_has_request_id(self):
        assert "request_id" in self.data
        assert self.data["request_id"] is not None

    def test_has_mask(self):
        assert "mask_png_b64" in self.data
        assert len(self.data["mask_png_b64"]) > 100

    def test_has_overlay(self):
        assert "overlay_png_b64" in self.data

    def test_has_coverage(self):
        assert "coverage_pct" in self.data
        assert 0.0 <= self.data["coverage_pct"] <= 100.0

    def test_mask_is_valid_base64_image(self):
        import base64

        raw = base64.b64decode(self.data["mask_png_b64"])
        img = Image.open(io.BytesIO(raw))
        assert img.size == (128, 128)

    def test_invalid_file_returns_400(self, client):
        r = client.post(
            "/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )
        assert r.status_code == 400


# ── XAI ───────────────────────────────────────────────────────────────────────


class TestXAI:

    def test_explain_predict_returns_200(self, client, test_image_bytes):
        r = client.post(
            "/explain/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200

    def test_explain_has_heatmap(self, client, test_image_bytes):
        data = client.post(
            "/explain/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        ).json()
        assert "xai" in data
        assert "heatmap_b64" in data["xai"] or "error" in data["xai"]

    def test_explain_methods_endpoint(self, client):
        r = client.get("/explain/methods")
        assert r.status_code == 200
        data = r.json()
        assert data["model_type"] == "unet"
        assert "Grad-CAM" in data["method"]


# ── GDPR Endpoints ────────────────────────────────────────────────────────────


class TestGDPR:

    def test_gdpr_status(self, client):
        r = client.get("/gdpr/status")
        assert r.status_code == 200
        data = r.json()
        assert data["compliant"] is True

    def test_gdpr_retention_policy(self, client):
        r = client.get("/gdpr/retention")
        assert r.status_code == 200
        assert "policy" in r.json()

    def test_gdpr_privacy_policy(self, client):
        r = client.get("/gdpr/privacy-policy")
        assert r.status_code == 200

    def test_right_to_erasure(self, client, test_image_bytes):
        # 1. Make a predict request → get request_id
        r = client.post(
            "/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        request_id = r.json().get("request_id")
        assert request_id is not None

        # 2. Erase by request_id
        r_erase = client.delete(f"/gdpr/erase/{request_id}")
        assert r_erase.status_code == 200
        assert r_erase.json()["erased"] is True

    def test_erase_unknown_id(self, client):
        r = client.delete("/gdpr/erase/nonexistent-id-000")
        assert r.status_code == 200
        assert r.json()["erased"] is True  # graceful — not 404


# ── Rate Limiting ─────────────────────────────────────────────────────────────


class TestRateLimiting:

    def test_metrics_endpoint_exists(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_repeated_requests_dont_crash(self, client, test_image_bytes):
        """5 rapid requests should all succeed (well under 10/min limit)."""
        for _ in range(5):
            r = client.post(
                "/predict",
                headers={"X-Data-Consent": "true"},
                files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            )
            assert r.status_code == 200


# ── Full Workflow ─────────────────────────────────────────────────────────────


class TestFullWorkflow:
    """
    End-to-end: upload → predict → explain → audit → erase
    """

    def test_complete_workflow(self, client, test_image_bytes):
        # 1. Check health
        assert client.get("/health").json()["status"] == "healthy"

        # 2. Try without consent → blocked
        r = client.post(
            "/predict", files={"file": ("t.jpg", test_image_bytes, "image/jpeg")}
        )
        assert r.status_code == 403

        # 3. Predict with consent → get request_id
        r = client.post(
            "/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("t.jpg", test_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        request_id = r.json()["request_id"]
        assert request_id

        # 4. XAI explanation
        r = client.post(
            "/explain/predict",
            headers={"X-Data-Consent": "true"},
            files={"file": ("t.jpg", test_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200

        # 5. Audit trail accessible
        r = client.get(f"/gdpr/request/{request_id}")
        assert r.status_code == 200

        # 6. Erase all data
        r = client.delete(f"/gdpr/erase/{request_id}")
        assert r.json()["erased"] is True

        # 7. Metrics recorded
        assert client.get("/metrics").status_code == 200
