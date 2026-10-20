"""Coverage for the traffic-law browse/search endpoints and corpus listing."""
import json

import pytest

from src.api.dependencies import get_settings


@pytest.fixture
def corpus(tmp_path):
    """Minimal two-article corpus exercising natural section ordering."""
    parents = {
        'VTL-1111-a': {'id': 'VTL-1111-a', 'article': '24', 'section': '1111-a',
                       'title': 'Owner liability', 'category': 'enforcement_program',
                       'refs': [], 'source': 'raw/a.txt', 'version': 1,
                       'text': 'Owner liability for failure to comply with signals.'},
        'VTL-1110': {'id': 'VTL-1110', 'article': '24', 'section': '1110',
                     'title': 'Obedience to traffic-control devices', 'category': 'core',
                     'refs': [], 'source': 'raw/b.txt', 'version': 1,
                     'text': 'Every person shall obey the instructions of any official traffic-control device.'},
        'VTL-1180': {'id': 'VTL-1180', 'article': '29', 'section': '1180',
                     'title': 'Speed restrictions', 'category': 'core',
                     'refs': [], 'source': 'raw/c.txt', 'version': 1,
                     'text': 'No person shall drive a vehicle at a speed greater than is reasonable and prudent.'},
        'VTL-1111': {'id': 'VTL-1111', 'article': '24', 'section': '1111',
                     'title': 'Traffic-control signal indications', 'category': 'core',
                     'refs': [], 'source': 'raw/d.txt', 'version': 1,
                     'text': 'Traffic-control signals indicating green, yellow and red shall apply to drivers.'},
    }
    children = [
        {'id': f'{pid}#full', 'parent_id': pid, 'article': p['article'], 'section': p['section'],
         'sub': 'full', 'category': p['category'], 'ctx': '', 'text': p['text']}
        for pid, p in parents.items()
    ]
    (tmp_path / 'parents.json').write_text(json.dumps(parents))
    (tmp_path / 'children.jsonl').write_text('\n'.join(map(json.dumps, children)))
    return tmp_path


def test_browse_lists_corpus_sorted_with_facets(test_client, corpus):
    settings = get_settings()
    settings.rag_mock = False
    settings.rag_out_dir = corpus
    response = test_client.get('/api/laws/')
    assert response.status_code == 200
    body = response.json()
    assert body['total'] == 4
    assert body['offset'] == 0
    # Natural order: 1110, 1111, 1111-a, then article 29
    assert [r['section_id'] for r in body['results']] == [
        'VTL-1110', 'VTL-1111', 'VTL-1111-a', 'VTL-1180']
    assert body['facets'] == {
        'categories': {'core': 3, 'enforcement_program': 1},
        'articles': {'24': 3, '29': 1}}
    # Full text and corpus metadata survive the round trip
    first = body['results'][0]
    assert 'traffic-control device' in first['content']
    assert first['metadata']['article'] == '24'
    assert first['metadata']['category'] == 'core'


def test_browse_filters_and_pagination(test_client, corpus):
    settings = get_settings()
    settings.rag_mock = False
    settings.rag_out_dir = corpus
    core = test_client.get('/api/laws/', params={'category': 'core'}).json()
    assert [r['section_id'] for r in core['results']] == ['VTL-1110', 'VTL-1111', 'VTL-1180']
    assert core['total'] == 3
    article_24 = test_client.get('/api/laws/', params={'article': '24'}).json()
    assert [r['section_id'] for r in article_24['results']] == [
        'VTL-1110', 'VTL-1111', 'VTL-1111-a']
    page = test_client.get('/api/laws/', params={'limit': 2, 'offset': 2}).json()
    assert [r['section_id'] for r in page['results']] == ['VTL-1111-a', 'VTL-1180']
    assert page['total'] == 4
    assert page['offset'] == 2


def test_search_with_query_still_uses_rag(test_client, corpus):
    settings = get_settings()
    settings.rag_mock = False
    settings.rag_out_dir = corpus
    settings.rag_mode = 'bm25'
    settings.rag_rerank = False
    response = test_client.get('/api/laws/', params={'q': 'speed reasonable and prudent'})
    assert response.status_code == 200
    body = response.json()
    assert body['query'] == 'speed reasonable and prudent'
    assert body['results'][0]['section_id'] == 'VTL-1180'
    assert body['results'][0]['score'] > 0


def test_search_rerank_is_opt_in_not_forced_by_settings(test_client, corpus, monkeypatch):
    """Interactive search must not inherit the slow cross-encoder pass
    unless the caller asks for it with ?rerank=true."""
    from src.api.services import law_service
    settings = get_settings()
    settings.rag_mock = False
    settings.rag_out_dir = corpus
    settings.rag_rerank = True
    seen = []
    monkeypatch.setattr(
        law_service, 'retrieve_laws',
        lambda q, out, s, rerank=None: seen.append(rerank) or [])
    assert test_client.get('/api/laws/', params={'q': 'yield'}).status_code == 200
    assert test_client.get('/api/laws/', params={'q': 'yield', 'rerank': 'true'}).status_code == 200
    assert seen == [False, True]


def test_single_law_by_id(test_client, corpus):
    settings = get_settings()
    settings.rag_mock = False
    settings.rag_out_dir = corpus
    found = test_client.get('/api/laws/VTL-1110')
    assert found.status_code == 200
    assert found.json()['title'] == 'Obedience to traffic-control devices'
    assert test_client.get('/api/laws/VTL-9999').status_code == 404


def test_browse_missing_corpus_is_actionable(test_client, tmp_path):
    settings = get_settings()
    settings.rag_mock = False
    settings.rag_out_dir = tmp_path / 'missing'
    response = test_client.get('/api/laws/')
    assert response.status_code == 503
    assert 'RAG_OUT_DIR' in response.json()['detail']


def test_mock_mode_browse_returns_demo_laws(test_client):
    response = test_client.get('/api/laws/')
    assert response.status_code == 200
    body = response.json()
    assert body['total'] >= 1
    assert all(r['section_id'].startswith('VTL-') for r in body['results'])
