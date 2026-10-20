"""Coverage for segmentation overlay generation, formatting, and serving."""
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_segmentation_overlays_are_generated_and_reported(monkeypatch, tmp_path):
    """analyze_case must receive a debug dir; existing overlay files must be
    returned per-photo, and photos without overlays must be skipped."""
    from src.car_damage_cost import cost_engine, pipeline, yolo_wrapper
    from src.api.services.damage_service import run_damage_assessment
    from src.api.config import Settings

    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_analyze_case(paths, damage_model, parts_model, vlm, cfg,
                          case_id=None, debug_dir=None):
        seen['debug_dir'] = Path(debug_dir)
        seen['case_id'] = case_id
        # Simulate viz.save_debug output — only for the first photo, so the
        # "photo without overlays" branch is covered too.
        first = Path(paths[0]).stem
        out = Path(debug_dir)
        out.mkdir(parents=True, exist_ok=True)
        for view in ('parts', 'damage', 'combined'):
            (out / f'{first}_{view}.jpg').write_bytes(b'\xff\xd8fake')
        return SimpleNamespace(fused=[], warnings=['no damage detected in any photo'])

    def fake_cost(*args, **kwargs):
        return SimpleNamespace(model_dump=lambda: {
            'total_min': 100.0, 'total_max': 200.0, 'currency': 'EGP',
            'lines': [], 'unpriced': [],
        })

    monkeypatch.setattr(pipeline, 'analyze_case', fake_analyze_case)
    monkeypatch.setattr(yolo_wrapper, 'YoloSegWrapper', lambda *a, **k: object())
    monkeypatch.setattr(cost_engine, 'load_tables', lambda p: {})
    monkeypatch.setattr(cost_engine, 'estimate_cost', fake_cost)

    photos = [str(tmp_path / 'a_photo.jpg'), str(tmp_path / 'b_photo.jpg')]
    report = run_damage_assessment(
        image_paths=photos,
        vehicle_info={'make': 'Toyota', 'model': 'Camry', 'year': 2020},
        assessment_id='DMG-TEST',
        settings=Settings(_env_file=None, damage_mock=False),
    )

    assert seen['debug_dir'] == Path('out') / 'damage' / 'DMG-TEST' / 'segmentation'
    assert seen['case_id'] == 'DMG-TEST'
    seg = report['segmentation_images']
    assert len(seg) == 1  # b_photo.jpg produced no overlays -> omitted
    entry = seg[0]
    assert entry['photo'] == 'a_photo.jpg'
    for view in ('parts', 'damage', 'combined'):
        assert Path(entry[view]).name == f'a_photo_{view}.jpg'
        assert entry[view].endswith(f'{view}.jpg')


def test_mock_assessment_reports_empty_segmentation(monkeypatch, tmp_path):
    from src.api.services.damage_service import run_damage_assessment
    from src.api.config import Settings
    report = run_damage_assessment(['photo.jpg'], {'make': 'T', 'model': 'C', 'year': 2020},
                                    'DMG-MOCK', Settings(_env_file=None, damage_mock=True))
    assert report['segmentation_images'] == []


def test_format_damage_record_exposes_segmentation_urls():
    """Stored overlay paths must surface as camelCase URL entries."""
    from src.api.routers.damage import _format_damage_record
    record = {
        'id': 'DMG-ABC123',
        'vehicle_info': {'make': 'Toyota', 'model': 'Camry', 'year': 2020},
        'images': ['/api/damage/images/DMG-ABC123/images/a.jpg'],
        'cost_report': {
            'lines': [], 'total_min': 100.0, 'total_max': 200.0, 'currency': 'EGP',
            'segmentation_images': [
                {'photo': 'a.jpg',
                 'parts': 'out/damage/DMG-ABC123/segmentation/a_parts.jpg',
                 'damage': 'out/damage/DMG-ABC123/segmentation/a_damage.jpg',
                 'combined': 'out/damage/DMG-ABC123/segmentation/a_combined.jpg'},
                {'photo': 'unreadable.jpg'},  # no views -> dropped
            ],
        },
    }
    formatted = _format_damage_record(record)
    seg = formatted['segmentationImages']
    assert len(seg) == 1
    assert seg[0]['photo'] == 'a.jpg'
    base = '/api/damage/images/DMG-ABC123/segmentation/a'
    assert seg[0]['parts'] == f'{base}_parts.jpg'
    assert seg[0]['damage'] == f'{base}_damage.jpg'
    assert seg[0]['combined'] == f'{base}_combined.jpg'


def test_config_resolves_prices_path_to_existing_file(monkeypatch):
    """from_env must find the shipped price tables (repo root has no
    data/prices.json — they live in the package's own data/ directory)."""
    from src.car_damage_cost.config import Config

    monkeypatch.delenv("PRICES_PATH", raising=False)
    prices = Path(Config.from_env().prices_path)
    assert prices.is_file(), f"{prices} does not exist"
    assert prices.name == "prices.json"


def test_upload_fallback_still_surfaces_real_overlays(test_client, tmp_path, monkeypatch):
    """If analysis wrote overlays but a later step failed, the fallback
    response must still return the real overlay files (never none)."""
    from src.api.services import damage_service

    monkeypatch.chdir(tmp_path)

    def fail_after_analysis(image_paths, vehicle_info, assessment_id, settings):
        seg = tmp_path / "out" / "damage" / assessment_id / "segmentation"
        seg.mkdir(parents=True, exist_ok=True)
        stem = Path(image_paths[0]).stem
        for view in ("parts", "damage", "combined"):
            (seg / f"{stem}_{view}.jpg").write_bytes(b"\xff\xd8" + view.encode())
        raise RuntimeError("Damage analysis failed: pricing exploded")

    monkeypatch.setattr(damage_service, "run_damage_assessment", fail_after_analysis)

    resp = test_client.post(
        "/api/damage/upload",
        files=[("images", ("car.jpg", b"\xff\xd8\xd9fake", "image/jpeg"))],
        data={"make": "Toyota", "model": "Camry", "year": "2019"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert any("Processed with fallback" in note for note in body["estimate"]["limitations"])

    seg = body["segmentationImages"]
    assert len(seg) == 1
    assert seg[0]["photo"].endswith("car.jpg")
    for view in ("parts", "damage", "combined"):
        assert seg[0][view].endswith(f"_{view}.jpg")
        served = test_client.get(seg[0][view])
        assert served.status_code == 200, view
        assert served.content.startswith(b"\xff\xd8"), view


def test_legacy_records_without_segmentation_format_cleanly():
    from src.api.routers.damage import _format_damage_record
    formatted = _format_damage_record({
        'id': 'DMG-OLD', 'vehicle_info': {}, 'images': [],
        'cost_report': {'lines': [], 'total_min': 10.0, 'total_max': 20.0},
    })
    assert formatted['segmentationImages'] == []


def test_serve_segmentation_and_uploaded_images(test_client, tmp_path, monkeypatch):
    """Bare filenames resolve to images/; segmentation/ files are served;
    unknown subdirectories and traversal attempts are rejected."""
    monkeypatch.chdir(tmp_path)
    seg_dir = tmp_path / 'out' / 'damage' / 'DMG-SEG' / 'segmentation'
    seg_dir.mkdir(parents=True)
    (seg_dir / 'a_parts.jpg').write_bytes(b'\xff\xd8segdata')
    img_dir = tmp_path / 'out' / 'damage' / 'DMG-SEG' / 'images'
    img_dir.mkdir(parents=True)
    (img_dir / 'a.jpg').write_bytes(b'\xff\xd8origdata')

    ok_seg = test_client.get('/api/damage/images/DMG-SEG/segmentation/a_parts.jpg')
    assert ok_seg.status_code == 200
    assert ok_seg.content == b'\xff\xd8segdata'

    ok_img = test_client.get('/api/damage/images/DMG-SEG/a.jpg')
    assert ok_img.status_code == 200
    assert ok_img.content == b'\xff\xd8origdata'

    assert test_client.get('/api/damage/images/DMG-SEG/other/a.jpg').status_code == 403
    assert test_client.get('/api/damage/images/DMG-SEG/segmentation/missing.jpg').status_code == 404

    # Dot-segment traversal is blocked by the handler itself (HTTP clients
    # normalize such URLs before sending, so exercise the guard directly).
    from fastapi import HTTPException
    from src.api.routers.damage import serve_damage_image
    for bad in ('images/../segmentation/a.jpg', '../DMG-SEG/images/a.jpg',
                '../../etc/passwd', '/etc/passwd'):
        with pytest.raises(HTTPException) as exc:
            serve_damage_image('DMG-SEG', bad)
        assert exc.value.status_code == 403, bad
