"""Verify the real SDK's URL construction without making provider requests."""
import json

import httpx
import openai
import pytest

from src.reasoning.reasoning_accident import _resolve_provider, analyze_accident


@pytest.mark.parametrize('base_url,expected_url,expected_model', [
    ('https://openrouter.ai/api/v1', 'https://openrouter.ai/api/v1/chat/completions', 'openai/gpt-4o-mini'),
    (' https://openrouter.ai/api/v1/chat/completions/ ', 'https://openrouter.ai/api/v1/chat/completions', 'openai/gpt-4o-mini'),
    ('https://api.groq.com/openai/v1/chat/completions', 'https://api.groq.com/openai/v1/chat/completions', 'gpt-4o-mini'),
    ('https://provider.invalid/custom/v1/', 'https://provider.invalid/custom/v1/chat/completions', 'gpt-4o-mini'),
])
def test_sdk_requests_chat_endpoint_exactly_once(monkeypatch, base_url, expected_url, expected_model):
    requests = []
    report = {'overall_assessment': 'No incident evidence supplied.'}

    def handler(request):
        requests.append(request)
        assert str(request.url) == expected_url
        payload = json.loads(request.content)
        assert payload['model'] == expected_model
        assert payload['response_format'] == {'type': 'json_object'}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(report)}}]})

    factory = openai.OpenAI
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: factory(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    assert analyze_accident('Validation only', {}, [], api_key='test-key', base_url=base_url) == report
    assert len(requests) == 1


def test_provider_resolution_preserves_explicit_models_and_matches_exact_host():
    assert _resolve_provider('sk-or-v1-test', '', 'anthropic/test') == (
        'https://openrouter.ai/api/v1', 'anthropic/test')
    assert _resolve_provider('test', '', 'custom') == (None, 'custom')
    assert _resolve_provider('test', 'https://provider.invalid/openrouter.ai/v1', 'custom') == (
        'https://provider.invalid/openrouter.ai/v1', 'custom')


def test_404_reports_configuration_error_without_exposing_credentials(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={'error': {'message': 'Not Found', 'code': 404}})

    factory = openai.OpenAI
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: factory(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    with pytest.raises(RuntimeError, match='Verify OPENAI_BASE_URL') as error:
        analyze_accident('Validation only', {}, [], api_key='secret-test-key', base_url='https://provider.invalid/v1')
    assert 'REASONING_MODEL' in str(error.value)
    assert 'secret-test-key' not in str(error.value)
