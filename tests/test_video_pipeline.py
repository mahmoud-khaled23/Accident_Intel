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


@pytest.mark.parametrize('backend', ['local', 'openai'])
def test_real_adapters_pass_video_tracking_and_context_through_models(test_client, video_bytes, monkeypatch, backend):
    from src.video_understanding import backend as selection
    monkeypatch.setattr(selection, 'cuda_available', lambda: backend == 'local')
    settings = get_settings()
    for name in ('tracking', 'vlm', 'rag', 'reasoning'):
        monkeypatch.setattr(settings, f'{name}_mock', False)
    monkeypatch.setattr(settings, 'video_vlm_api', backend)
    monkeypatch.setattr(settings, 'vlm_model', 'test-vision')
    monkeypatch.setattr(settings, 'vlm_api_key', 'test-key')
    monkeypatch.setattr(settings, 'openai_api_key', 'test-key')
    monkeypatch.setattr(settings, 'vlm_n_frames', 3)
    seen = {}
    stages = []
    from src.vehicle_tracking import tracker

    def process_video(video_path, output_video, output_json, model_path):
        assert Path(video_path).read_bytes() == video_bytes
        assert str(settings.data_dir) in output_json
        seen['tracking_model'] = model_path
        stages.append('tracking')
        return {7: {'class': 'car', 'trajectory': {'0': {'box': [0, 30, 25, 60], 'confidence': 0.83},
                                                   '10': {'box': [40, 30, 65, 60], 'confidence': 0.91}}}}
    monkeypatch.setattr(tracker, 'process_video', process_video)

    description = 'Vehicle 7 moves through the frame. No collision is visible.'
    laws = [{'id': 'VTL-TEST', 'title': 'Test passage', 'text': 'A test passage.', 'score': 0.04}]

    if backend == 'local':
        import torch
        from transformers import BatchEncoding
        from src.video_understanding import analyze_video as local_vlm
        from src.video_understanding.prompts import SYSTEM_PROMPT
        from src.api.services import vlm_service

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                assert messages[0]['content'] == SYSTEM_PROMPT
                video, prompt = messages[-1]['content']
                assert Path(video['video']).read_bytes() == video_bytes
                assert str(settings.data_dir) in video['video']
                assert video['nframes'] == 2  # Qwen rounds to an even count.
                assert '"id":7' in prompt['text']
                return 'Video prompt'

            def __call__(self, **kwargs):
                # Real Qwen video preprocessing decoded the actual uploaded clip.
                assert len(kwargs['videos']) == 1
                assert kwargs['videos'][0].shape[0] == 2
                return BatchEncoding({'input_ids': torch.tensor([[1, 2]])})

            def batch_decode(self, tokens, **kwargs):
                assert tokens[0].tolist() == [3]
                return [description]

        class Model:
            device = 'cpu'

            def generate(self, **kwargs):
                assert kwargs['max_new_tokens'] == settings.vlm_max_tokens
                assert kwargs['do_sample'] is False
                seen['vlm'] = True
                stages.append('vlm')
                return torch.tensor([[1, 2, 3]])

        def load_model(model_id):
            assert model_id == settings.vlm_model_id
            return Model(), Processor()

        monkeypatch.setattr(local_vlm, 'load_model', load_model)
        for name in ('_vlm_model', '_vlm_processor', '_vlm_model_id'):
            monkeypatch.setattr(vlm_service, name, None)

    def retrieve_for_report(**kwargs):
        assert kwargs['description'] == description
        seen['rag'] = kwargs
        stages.append('rag')
        return laws
    monkeypatch.setitem(sys.modules, 'src.rag.retrieval.report_retrieval', SimpleNamespace(retrieve_for_report=retrieve_for_report))

    class ModelClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def create(self, **kwargs):
            content = kwargs['messages'][-1]['content']
            if isinstance(content, list):
                assert backend == 'openai', 'Local video inference must not call a hosted vision API'
                images = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
                assert len(images) == settings.vlm_max_images and len(set(images)) == settings.vlm_max_images
                assert all(x.startswith('data:image/jpeg;base64,') for x in images)
                assert '"id":7' in content[0]['text']
                assert '1.100 seconds' in str(content)
                seen['vlm'] = True
                stages.append('vlm')
                result = description
            else:
                assert description in content and 'VTL-TEST' in content and '"id": 7' in content
                assert 'Review the green vehicle' in content
                seen['reasoning'] = True
                stages.append('reasoning')
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
    assert stages == ['tracking', 'vlm', 'rag', 'reasoning']
    assert test_client.get(f'/api/accidents/{aid}/vlm').json()['description'] == description
    assert test_client.get(f'/api/accidents/{aid}/laws').json()['laws'][0]['section_id'] == 'VTL-TEST'
    assert test_client.get(f'/api/accidents/{aid}/report').json()['report'] == result['report']


def test_video_backend_is_independent_of_shared_vlm_api(monkeypatch):
    from src.api.config import Settings
    from src.api.services.vlm_service import backend_info
    from src.video_understanding import backend
    monkeypatch.setattr(backend, 'cuda_available', lambda: True)

    monkeypatch.delenv('VIDEO_VLM_API', raising=False)
    monkeypatch.setenv('VLM_API', 'openai')
    monkeypatch.setenv('VLM_MOCK', 'false')
    monkeypatch.setenv('VLM_MODEL_ID', 'Qwen/Qwen2-VL-7B-Instruct')
    info = backend_info(Settings(_env_file=None))
    assert info['backend'] == 'local'
    assert info['requested'] == 'auto'
    assert info['module'] == 'src.video_understanding.analyze_video'
    monkeypatch.setenv('VIDEO_VLM_API', 'openai')
    assert backend_info(Settings(_env_file=None))['module'] == 'src.video_understanding.hosted'


def test_health_identifies_selected_video_model(test_client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, 'vlm_mock', False)
    monkeypatch.setattr(settings, 'video_vlm_api', 'local')
    info = test_client.get('/api/health').json()['video_vlm']
    assert info['backend'] == 'local'
    assert info['model'] == settings.vlm_model_id
    assert info['module'] == 'src.video_understanding.analyze_video'


def test_local_vlm_uses_own_prompts_despite_bare_import_collisions(monkeypatch):
    import importlib
    from src.video_understanding import analyze_video, prompts

    monkeypatch.setitem(sys.modules, 'prompts', SimpleNamespace(
        SYSTEM_PROMPT='unrelated', TRAFFIC_ANALYSIS_PROMPT='unrelated', build_accident_prompt=None,
    ))
    module = importlib.reload(analyze_video)
    assert module.SYSTEM_PROMPT == prompts.SYSTEM_PROMPT
    assert module.TRAFFIC_ANALYSIS_PROMPT == prompts.TRAFFIC_ANALYSIS_PROMPT


def test_empty_vlm_output_stops_before_rag_and_reasoning(test_client, video_bytes, monkeypatch):
    from src.api.services import vlm_service, law_service, reasoning_service

    monkeypatch.setattr(vlm_service, 'analyze_video', lambda *args: '  \n')

    def must_not_run(*args):
        pytest.fail('Downstream stages must not run without a VLM description')

    monkeypatch.setattr(law_service, 'retrieve_laws', must_not_run)
    monkeypatch.setattr(reasoning_service, 'analyze_accident', must_not_run)
    aid = upload(test_client, video_bytes).json()['id']
    test_client.post(f'/api/accidents/{aid}/analyze')
    result = test_client.get(f'/api/accidents/{aid}').json()
    assert result['status'] == 'failed' and result['failed_stage'] == 'vlm_analysis'
    assert result['laws'] is None and result['report'] is None


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
                assert sheet.size == (384, 266)
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
