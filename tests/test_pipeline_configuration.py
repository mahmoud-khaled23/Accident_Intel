"""Regression coverage for renamed modules and real backend selection."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from src.api.config import PROJECT_ROOT, Settings
from src.api.dependencies import get_settings
from src.video_understanding import backend


@pytest.mark.parametrize('gpu,expected', [(True, 'local'), (False, 'openai')])
def test_auto_uses_gpu_or_api(monkeypatch, gpu, expected):
    monkeypatch.setattr(backend, 'cuda_available', lambda: gpu)
    assert backend.select_backend('auto') == expected
    assert backend.select_backend('openai') == 'openai'
    assert backend.select_backend('local') == 'local'


def test_settings_are_independent_of_cwd_and_damage_credentials(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DATA_DIR', 'data/runtime')
    monkeypatch.delenv('DB_PATH', raising=False)
    monkeypatch.setenv('RAG_OUT_DIR', 'data/traffic_laws/chunks')
    monkeypatch.setenv('TRACKING_MODEL_PATH', 'weights\\vehicle_tracking\\yolo11x.pt')
    monkeypatch.setenv('VLM_API_KEY', 'damage-key')
    monkeypatch.setenv('VIDEO_VLM_API_KEY', 'video-key')
    settings = Settings(_env_file=None)
    assert settings.data_dir == PROJECT_ROOT / 'data/runtime'
    assert settings.db_path == settings.data_dir / 'accident_intel.db'
    assert settings.rag_out_dir == PROJECT_ROOT / 'data/traffic_laws/chunks'
    assert settings.tracking_model_path == str(PROJECT_ROOT / 'weights/vehicle_tracking/yolo11x.pt')
    assert settings.vlm_api_key == 'video-key'


def test_upload_backend_is_persisted_and_retry_can_change_it(test_client, video_bytes, monkeypatch):
    from src.api.services import vlm_service
    selected = []
    monkeypatch.setattr(vlm_service, 'analyze_video', lambda path, aid, settings, tracks: selected.append(settings.video_vlm_api) or 'Observed clip.')
    response = test_client.post('/api/accidents/upload', files={'file': ('clip.mp4', video_bytes)},
                                data={'video_vlm_api': 'openai'})
    assert response.status_code == 201
    aid = response.json()['id']
    assert response.json()['video_vlm_api'] == 'openai'
    test_client.post(f'/api/accidents/{aid}/analyze')
    test_client.post(f'/api/accidents/{aid}/analyze?video_vlm_api=local')
    assert selected == ['openai', 'local']
    assert test_client.get(f'/api/accidents/{aid}').json()['video_vlm_api'] == 'local'
    assert test_client.post(f'/api/accidents/{aid}/analyze?video_vlm_api=invalid').status_code == 422


def test_local_without_gpu_fails_with_partial_tracking(test_client, video_bytes, monkeypatch):
    monkeypatch.setattr(backend, 'cuda_available', lambda: False)
    monkeypatch.setattr(get_settings(), 'vlm_mock', False)
    aid = test_client.post('/api/accidents/upload', files={'file': ('clip.mp4', video_bytes)},
                           data={'video_vlm_api': 'local'}).json()['id']
    test_client.post(f'/api/accidents/{aid}/analyze')
    result = test_client.get(f'/api/accidents/{aid}').json()
    assert result['status'] == 'failed'
    assert result['failed_stage'] == 'vlm_analysis'
    assert 'CUDA GPU' in result['error']
    assert result['tracking_data'] and result['report'] is None


def test_auto_cpu_with_missing_api_credentials_never_uses_demo(monkeypatch, tmp_path):
    from src.api.services.vlm_service import analyze_video
    monkeypatch.setattr(backend, 'cuda_available', lambda: False)
    settings = Settings(_env_file=None, vlm_mock=False, video_vlm_api='auto', vlm_model='', vlm_api_key='')
    with pytest.raises(RuntimeError, match='VIDEO_VLM_MODEL'):
        analyze_video(str(tmp_path / 'clip.mp4'), 'test', settings)


def test_package_imports_ignore_colliding_bare_modules(monkeypatch):
    for name in ('config', 'schemas', 'prompts', 'pipeline'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    from src.car_damage_cost import config, cost_engine
    from src.api.services.vlm_service import _analysis_prompt
    from src.video_understanding.prompts import TRAFFIC_ANALYSIS_PROMPT
    importlib.reload(cost_engine)
    assert config.Config.__module__ == 'src.car_damage_cost.config'
    assert _analysis_prompt(None) == TRAFFIC_ANALYSIS_PROMPT


def test_real_bm25_retrieval_accepts_string_paths_and_refreshes_corpus(tmp_path):
    from src.rag.retrieval.index_and_retrieve import retrieve, load
    from src.rag.retrieval.report_retrieval import retrieve_for_report
    parents = {
        f'VTL-{i}': {'id': f'VTL-{i}', 'title': word, 'text': word, 'category': 'core', 'refs': []}
        for i, word in enumerate(['yield intersection', 'signal red', 'parking curb'])
    }
    children = [{'id': f'{p["id"]}-a', 'parent_id': p['id'], 'text': p['text'], 'category': 'core'} for p in parents.values()]
    (tmp_path / 'parents.json').write_text(json.dumps(parents))
    (tmp_path / 'children.jsonl').write_text('\n'.join(map(json.dumps, children)))
    assert retrieve(['yield'], str(tmp_path), mode='bm25', rerank=False)[0]['id'] == 'VTL-0'
    assert retrieve_for_report('parking', str(tmp_path), mode='bm25')[0]['id'] == 'VTL-2'
    from src.api.services.law_service import get_law_by_id
    settings = Settings(_env_file=None, rag_mock=False)
    assert get_law_by_id('0', str(tmp_path), settings)['section_id'] == 'VTL-0'
    assert get_law_by_id('999', str(tmp_path), settings) is None
    children[0]['text'], children[2]['text'] = children[2]['text'], children[0]['text']
    (tmp_path / 'children.jsonl').write_text('\n'.join(map(json.dumps, children)))
    assert retrieve(['yield'], str(tmp_path), mode='bm25')[0]['id'] == 'VTL-2'
    with pytest.raises(FileNotFoundError, match='RAG_OUT_DIR'):
        load(tmp_path / 'missing')


def test_frame_sampling_does_not_duplicate_short_videos(video_bytes, tmp_path):
    from src.video_understanding.contact_sheet import sample_frames
    video = tmp_path / 'clip.mp4'
    video.write_bytes(video_bytes)
    frames = sample_frames(str(video), 32)
    assert len(frames) == 12
    assert len({timestamp for timestamp, _ in frames}) == 12


def test_braking_alone_does_not_implicate_unrelated_vehicle():
    from src.vehicle_tracking.accident_candidates import identify_accident_candidates
    tracks = {}
    for track_id, y in [(1, 0), (2, 1000)]:
        tracks[track_id] = {'class': 'car', 'trajectory': {
            str(i): {'center': [x, y], 'box': [x, y, x + 10, y + 10]}
            for i, x in enumerate([0, 10, 11, 12])
        }}
    assert identify_accident_candidates(tracks) == []


@pytest.mark.parametrize('field', ['vehicle', 'law'])
def test_reasoning_rejects_invented_references(monkeypatch, field):
    from src.api.services.reasoning_service import analyze_accident
    from src.reasoning import reasoning_accident
    report = {'overall_assessment': 'Assessment', 'vehicles': [], 'violations': []}
    if field == 'vehicle':
        report['vehicles'] = [{'id': 999, 'role': 'primary_at_fault'}]
    else:
        report['violations'] = [{'law_section': 'VTL-999', 'description': 'Invented citation'}]
    monkeypatch.setattr(reasoning_accident, 'analyze_accident', lambda *a, **kw: report)
    with pytest.raises(RuntimeError, match='Invalid reasoning model response'):
        analyze_accident('Observed scene', {'trajectories': []}, [], Settings(_env_file=None, reasoning_mock=False))


def test_law_corpus_error_is_actionable_in_api(test_client, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, 'rag_mock', False)
    monkeypatch.setattr(settings, 'rag_out_dir', tmp_path / 'missing')
    response = test_client.get('/api/laws/?q=yield')
    assert response.status_code == 503
    assert 'RAG_OUT_DIR' in response.json()['detail']


def test_reasoning_authentication_error_explains_configuration(monkeypatch):
    import httpx
    import openai
    from src.reasoning.reasoning_accident import analyze_accident
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
        def close(self): pass
        def create(self, **kwargs):
            raise openai.AuthenticationError('User not found', response=httpx.Response(401, request=httpx.Request('POST', 'https://provider.invalid')), body={})
    monkeypatch.setattr(openai, 'OpenAI', Client)
    with pytest.raises(RuntimeError, match='Verify OPENAI_API_KEY and OPENAI_BASE_URL'):
        analyze_accident('Scene', {}, [], api_key='invalid')


def test_damage_inference_failure_never_returns_fake_prices(monkeypatch):
    from src.api.services.damage_service import run_damage_assessment
    from src.car_damage_cost import pipeline, yolo_wrapper
    def fail(**kwargs):
        raise RuntimeError('Model unavailable')
    monkeypatch.setattr(pipeline, 'analyze_case', fail)
    monkeypatch.setattr(yolo_wrapper, 'YoloSegWrapper', lambda *args: object())
    with pytest.raises(RuntimeError, match='Damage analysis failed: Model unavailable'):
        run_damage_assessment(['photo.jpg'], {'make': 'Test', 'model': 'Test', 'year': 2020}, 'test', Settings(_env_file=None, damage_mock=False))
