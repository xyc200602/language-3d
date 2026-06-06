"""Tests for complex robot design endpoints (Task 58)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from lang3d.web.app import app


class TestDesignEndpoints:
    """Test the 4 design API endpoints."""

    def test_app_exists(self):
        assert app is not None
        assert app.title == "Language-3D Agent Monitor"

    def test_design_hierarchy_endpoint(self):
        client = TestClient(app)
        response = client.get("/api/design/hierarchy")
        assert response.status_code == 200
        data = response.json()
        assert "subsystems" in data
        assert len(data["subsystems"]) == 5

    def test_design_hierarchy_total_parts(self):
        client = TestClient(app)
        response = client.get("/api/design/hierarchy")
        data = response.json()
        assert data["total_parts"] == 41

    def test_design_assembly_tree_endpoint(self):
        client = TestClient(app)
        response = client.get("/api/design/assembly-tree")
        assert response.status_code == 200
        data = response.json()
        assert "tree" in data

    def test_design_assembly_tree_root(self):
        client = TestClient(app)
        response = client.get("/api/design/assembly-tree")
        data = response.json()
        tree = data["tree"]
        assert tree["name"] == "base_plate"

    def test_design_stability_endpoint(self):
        client = TestClient(app)
        response = client.get("/api/design/stability")
        assert response.status_code == 200
        data = response.json()
        assert "total_mass_kg" in data
        assert "center_of_mass_mm" in data
        assert "static_stability" in data
        assert "tip_over_risk" in data

    def test_design_stability_has_risk_level(self):
        client = TestClient(app)
        response = client.get("/api/design/stability")
        data = response.json()
        risk = data.get("tip_over_risk", {})
        assert "risk_level" in risk
        assert risk["risk_level"]  # non-empty

    def test_design_power_budget_endpoint(self):
        client = TestClient(app)
        response = client.get("/api/design/power-budget")
        assert response.status_code == 200
        data = response.json()
        assert "consumers" in data
        assert "peak_power_w" in data
        assert "avg_power_w" in data
        assert len(data["consumers"]) > 0

    def test_design_power_budget_positive(self):
        client = TestClient(app)
        response = client.get("/api/design/power-budget")
        data = response.json()
        assert data["peak_power_w"] > 0
        assert data["avg_power_w"] > 0


class TestDesignTabUI:
    """Test that index.html includes the Design tab."""

    def test_index_html_has_design_tab(self):
        html_path = Path(__file__).resolve().parents[1] / "src" / "lang3d" / "web" / "static" / "index.html"
        if not html_path.exists():
            # Fallback: try to get from the web endpoint
            client = TestClient(app)
            response = client.get("/")
            assert response.status_code == 200
            content = response.text
        else:
            content = html_path.read_text(encoding="utf-8")
        assert 'data-tab="design"' in content
        assert "panel-design" in content
