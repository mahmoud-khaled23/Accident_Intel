"""Upload, real adapter contracts, failure/retry, and concurrency regressions."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.api.dependencies import get_settings
from src.api.models import claim_analysis, get_accident, update_accident, recover_interrupted_analyses


def upload(client, content, **data):
    return client.post('/api/accidents/upload', files={'file': ('clip.mp4', content, 'video/mp4')}, data=data)


def test_upload_metadata_playback_and_range(test_client, video_bytes):
    response = upload(test_client, video_bytes, location='Test junction', incident_date='2026-10-13T10:00:00Z')
    assert response.status_code == 201
    record = response.json()
    assert record['video_filename'] == 'clip.mp4'
    assert record['location'] == 'Test junction'
    assert record['video_metadata'] == {'width': 160, 'height': 96, 'fps': 10.0, 'frame_count': 12, 'duration_seconds': 1.2}
    video = test_client.get(f"/api/accidents/{record['id']}/video")
    assert video.content == video_bytes
    part = test_client.get(f"/api/accidents/{record['id']}/video", headers={'Range': 'bytes=0-31'})
    assert part.status_code == 206
    assert part.content == video_bytes[:32]


@pytest.mark.parametrize('content', [b'', b'not a video'])
def test_rejects_invalid_video_and_cleans_storage(test_client, content):
    assert upload(test_client, content).status_code == 400
    assert test_client.get('/api/accidents/').json()['total'] == 0
    assert not list((get_settings().data_dir / 'accidents').iterdir())


def test_format_size_and_pagination_validation(test_client, monkeypatch):
    assert test_client.post('/api/accidents/upload', files={'file': ('x.exe', b'bad')}).status_code == 415
    monkeypatch.setattr(get_settings(), 'max_upload_mb', 1)
    assert upload(test_client, b'0' * (1024 * 1024 + 1)).status_code == 413
    assert test_client.get('/api/accidents/?page_size=0').status_code == 422
    assert test_client.get('/api/accidents/?page=-1').status_code == 422


def test_real_adapters_pass_video_tracking_and_context_through_models(test_client, video_bytes, monkeypatch):
    settings = get_settings()
    for name in ('tracking', 'vlm', 'rag', 'reasoning'):
        monkeypatch.setattr(settings, f'{name}_mock', False)
    monkeypatch.setattr(settings, 'vlm_api', 'openai')
    monkeypatch.setattr(settings, 'vlm_model', 'test-vision')
    monkeypatch.setattr(settings, 'vlm_api_key', 'test-key')
    monkeypatch.setattr(settings, 'openai_api_key', 'test-key')
    monkeypatch.setattr(settings, 'vlm_n_frames', 3)
    seen = {}
    from vehicle_tracking import tracker

    def process_video(video_path, output_video, output_json, model_path):
        assert Path(video_path).read_bytes() == video_bytes
        assert str(settings.data_dir) in output_json
        seen['tracking_model'] = model_path
        return {7: {'class': 'car', 'trajectory': {'0': {'box': [0, 30, 25, 60], 'confidence': 0.83},
                                                   '10': {'box': [40, 30, 65, 60], 'confidence': 0.91}}}}
    monkeypatch.setattr(tracker, 'process_video', process_video)

    description = 'Vehicle 7 moves through the frame. No collision is visible.'
    laws = [{'id': 'VTL-TEST', 'title': 'Test passage', 'text': 'A test passage.', 'score': 0.04}]

    def retrieve_for_report(**kwargs):
        assert kwargs['description'] == description
        seen['rag'] = kwargs
        return laws
    monkeypatch.setitem(sys.modules, 'rag.retrieval.report_retrieval', SimpleNamespace(retrieve_for_report=retrieve_for_report))

    class ModelClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def create(self, **kwargs):
            content = kwargs['messages'][-1]['content']
            if isinstance(content, list):
                images = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
                assert len(images) == 3 and len(set(images)) == 3
                assert all(x.startswith('data:image/jpeg;base64,') for x in images)
                assert '"id": 7' in content[0]['text']
                assert '1.100 seconds' in str(content)
                seen['vlm'] = True
                result = description
            else:
                assert description in content and 'VTL-TEST' in content and '"id": 7' in content
                assert 'Review the green vehicle' in content
                seen['reasoning'] = True
                result = json.dumps({'overall_assessment': 'No observed collision.', 'overall_level': 'insufficient_evidence',
                                     'vehicles': [], 'violations': [], 'confidence': 0.8, 'limitations': ['Short clip.']})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])
    import openai
    monkeypatch.setattr(openai, 'OpenAI', ModelClient)
    record = upload(test_client, video_bytes, notes='Review the green vehicle').json()
    aid = record['id']
    assert test_client.post(f'/api/accidents/{aid}/analyze').status_code == 200
    result = test_client.get(f'/api/accidents/{aid}').json()
    assert result['status'] == 'completed', result.get('error')
    assert result['is_demo_data'] is False
    assert result['vlm_description'] == description
    assert result['tracking_data']['fps'] == 10
    assert result['tracking_data']['trajectories'][0]['frames'][1]['timestamp'] == 1
    assert result['tracking_data']['trajectories'][0]['frames'][0]['confidence'] == 0.83
    assert result['report']['overall_assessment'] == 'No observed collision.'
    assert set(seen) == {'tracking_model', 'vlm', 'rag', 'reasoning'}


def test_failure_is_visible_partial_results_survive_and_retry_clears_stale_data(test_client, video_bytes, monkeypatch):
    from src.api.services import vlm_service
    aid = upload(test_client, video_bytes).json()['id']
    original = vlm_service.analyze_video
    def fail(*args): raise RuntimeError('Model unavailable')
    monkeypatch.setattr(vlm_service, 'analyze_video', fail)
    test_client.post(f'/api/accidents/{aid}/analyze')
    result = test_client.get(f'/api/accidents/{aid}').json()
    assert result['status'] == 'failed'
    assert result['failed_stage'] == 'vlm_analysis'
    assert 'Model unavailable' in result['error']
    assert result['tracking_data'] is not None and result['report'] is None
    monkeypatch.setattr(vlm_service, 'analyze_video', original)
    test_client.post(f'/api/accidents/{aid}/analyze')
    result = test_client.get(f'/api/accidents/{aid}').json()
    assert result['status'] == 'completed' and result['error'] is None
    assert result['is_demo_data'] is True


def test_atomic_claim_blocks_duplicate_jobs_delete_and_replacement(test_client, video_bytes):
    aid = upload(test_client, video_bytes).json()['id']
    db = str(get_settings().db_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(executor.map(lambda _: claim_analysis(db, aid, False), range(8)))
    assert sum(claims) == 1
    assert test_client.post(f'/api/accidents/{aid}/analyze').status_code == 409
    assert test_client.delete(f'/api/accidents/{aid}').status_code == 409
    assert test_client.post(f'/api/accidents/{aid}/video', files={'file': ('new.mp4', video_bytes)}).status_code == 409


def test_restart_and_video_replacement(test_client, video_bytes):
    aid = upload(test_client, video_bytes).json()['id']
    db = str(get_settings().db_path)
    update_accident(db, aid, {'status': 'reasoning', 'report': {'old': True}})
    recover_interrupted_analyses(db)
    assert get_accident(db, aid)['status'] == 'failed'
    assert 'restart' in get_accident(db, aid)['error']
    response = test_client.post(f'/api/accidents/{aid}/video', files={'file': ('replacement.mp4', video_bytes)})
    assert response.status_code == 200
    record = test_client.get(f'/api/accidents/{aid}').json()
    assert record['video_filename'] == 'replacement.mp4'
    assert record['report'] is None and record['error'] is None
    assert test_client.delete(f'/api/accidents/{aid}').status_code == 204
    assert not (get_settings().data_dir / 'accidents' / aid).exists()


def test_hosted_vlm_contact_sheets_respect_image_limit(test_client, video_bytes, monkeypatch, tmp_path):
    from src.api.services.vlm_service import _analyze_via_openai_api
    import openai
    import base64
    import io
    from PIL import Image
    path = tmp_path / 'clip.mp4'
    path.write_bytes(video_bytes)
    settings = get_settings()
    monkeypatch.setattr(settings, 'vlm_n_frames', 12)
    monkeypatch.setattr(settings, 'vlm_max_images', 3)
    monkeypatch.setattr(settings, 'vlm_api_key', 'test-key')
    monkeypatch.setattr(settings, 'vlm_model', 'test-model')

    class Client:
        def __init__(self, **kwargs): self.chat = SimpleNamespace(completions=self)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def create(self, **kwargs):
            content = kwargs['messages'][-1]['content']
            images = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
            assert len(images) == 3
            for data in images:
                sheet = Image.open(io.BytesIO(base64.b64decode(data.split(',')[1])))
                assert sheet.size == (768, 482)
            captions = ' '.join(x['text'] for x in content[1:] if x['type'] == 'text')
            for i in range(12): assert f'{i / 10:.3f} seconds' in captions
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Observed frames.'))])
    monkeypatch.setattr(openai, 'OpenAI', Client)
    assert _analyze_via_openai_api(str(path), settings) == 'Observed frames.'


@pytest.mark.parametrize('report', [[], {'overall_assessment': ''},
    {'overall_assessment': 'Test', 'confidence': float('nan')},
    {'overall_assessment': 'Test', 'vehicles': [{'id': 1, 'violations': 'not a list'}]},
    {'overall_assessment': 'Test', 'overall_level': 'invented level'}])
def test_malformed_reasoning_results_fail_instead_of_crashing_frontend(test_client, monkeypatch, report):
    import openai
    from src.api.services.reasoning_service import analyze_accident
    settings = get_settings()
    monkeypatch.setattr(settings, 'reasoning_mock', False)
    monkeypatch.setattr(settings, 'openai_api_key', 'test-key')
    class Client:
        def __init__(self, **kwargs): self.chat = SimpleNamespace(completions=self)
        def close(self): pass
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(report)))])
    monkeypatch.setattr(openai, 'OpenAI', Client)
    with pytest.raises(RuntimeError, match='Invalid reasoning model response'):
        analyze_accident('Visible scene.', {'trajectories': []}, [], settings)
