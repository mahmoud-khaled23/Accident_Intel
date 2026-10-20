"""Response-contract tests for damage report formatting with missing data.

The API legitimately emits JSON ``null`` for optional vehicle fields the
user did not provide (``mileage``, ``vin``, ``repairLocation``).  The report
page must receive those as ``null`` — never coerced to 0 — and still get
fully numeric estimates.  Regression coverage for the report-page crash
``Cannot read properties of null (reading 'toLocaleString')``.
"""


def _record_with_null_optionals(**overrides):
    """Build a raw DB-shaped record whose optional vehicle fields are null."""
    record = {
        "id": "DMG-FMT1",
        "vehicle_info": {
            "make": "Toyota",
            "model": "Camry",
            "year": 2019,
            "color": "Silver",
            "mileage": None,
            "vin": None,
            "repair_location": None,
            "currency": "EGP",
        },
        "images": [],
        "cost_report": {
            "lines": [
                {
                    "part": "hood",
                    "action": "repair_and_paint",
                    "damages": ["dent x1 (minor)"],
                    "total_min": 1000.0,
                    "total_max": 1500.0,
                    "needs_review": False,
                }
            ],
            "total_min": 1000.0,
            "total_max": 1500.0,
            "currency": "EGP",
            "notes": ["Processed 1 photo"],
        },
    }
    record.update(overrides)
    return record


def test_format_report_with_null_optional_fields():
    """Null mileage/vin/repairLocation must stay null (not 0 / not dropped)
    while the numeric estimate is fully populated."""
    from src.api.routers.damage import _format_damage_record

    formatted = _format_damage_record(_record_with_null_optionals())

    vinfo = formatted["vehicleInfo"]
    assert vinfo["mileage"] is None
    assert vinfo["vin"] is None
    assert vinfo["repairLocation"] is None

    est = formatted["estimate"]
    assert est["subtotal"] == 1250.0
    assert est["lowEstimate"] == 1000.0
    assert est["highEstimate"] == 1500.0
    assert est["total"] == round(1250.0 * 1.14, 2)
    assert est["limitations"] == ["Processed 1 photo"]

    # Detected parts / repair items must survive the null optional fields
    assert formatted["damagedParts"][0]["name"] == "Hood"
    assert formatted["repairItems"][0]["total"] == 1250.0


def test_format_report_with_valid_numbers_preserves_them():
    """Reports with valid values must still format exactly as before."""
    from src.api.routers.damage import _format_damage_record

    record = _record_with_null_optionals(
        vehicle_info={
            "make": "Toyota",
            "model": "Camry",
            "year": 2019,
            "color": "Silver",
            "mileage": 62000,
            "vin": "4T1B11HK9LU000001",
            "repair_location": "Cairo",
        },
    )
    formatted = _format_damage_record(record)

    vinfo = formatted["vehicleInfo"]
    assert vinfo["mileage"] == 62000
    assert vinfo["vin"] == "4T1B11HK9LU000001"
    assert vinfo["repairLocation"] == "Cairo"
    assert formatted["estimate"]["total"] == 1425.0


def test_format_report_null_notes_become_empty_list():
    """A stored cost_report with notes=null must format as [] not null."""
    from src.api.routers.damage import _format_damage_record

    record = _record_with_null_optionals()
    record["cost_report"]["notes"] = None
    formatted = _format_damage_record(record)
    assert formatted["estimate"]["limitations"] == []


def test_upload_without_optional_fields_emits_json_nulls(test_client, monkeypatch):
    """Live contract: POST /upload without mileage/vin returns JSON nulls
    for those fields plus numeric estimates (the exact shape that crashed
    the report page before it learned to render N/A)."""
    from src.api.services import damage_service

    def fake_run(image_paths, vehicle_info, assessment_id, settings):
        return {
            "car_info": {"make": vehicle_info.get("make"), "model": vehicle_info.get("model"),
                         "year": vehicle_info.get("year"), "tier": "mid"},
            "total_min": 100.0,
            "total_max": 200.0,
            "currency": "EGP",
            "lines": [],
            "notes": [],
            "unpriced": [],
            "segmentation_images": [],
        }

    monkeypatch.setattr(damage_service, "run_damage_assessment", fake_run)

    resp = test_client.post(
        "/api/damage/upload",
        files=[("images", ("car.jpg", b"\xff\xd8\xd9fake", "image/jpeg"))],
        data={"make": "Toyota", "model": "Camry", "year": "2019"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # JSON null — the frontend renders these as "N/A"
    assert body["vehicleInfo"]["mileage"] is None
    assert body["vehicleInfo"]["vin"] is None
    assert body["vehicleInfo"]["repairLocation"] is None

    # Numeric fields stay real numbers
    assert body["estimate"]["total"] == 114.0  # 100 + 14% tax
    assert body["estimate"]["lowEstimate"] == 100.0
    assert body["estimate"]["highEstimate"] == 200.0
    assert isinstance(body["estimate"]["limitations"], list)
