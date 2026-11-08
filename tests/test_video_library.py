from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

from traffic_edge import web_ui


class VideoLibraryTests(unittest.TestCase):
    def test_original_red_light_examples_remain_primary_and_ambiguous_crashes_are_withdrawn(self):
        # Catalogue policy must be testable without redistributing source videos.
        with patch.object(Path, 'is_file', return_value=True):
            items = {item['id']: item for item in web_ui.load_evaluation_candidates()}
        original = json.loads(web_ui.CANDIDATE_CATALOG_PATH.read_text(encoding='utf-8'))['candidates']
        self.assertEqual(len(original), 3)
        for record in original:
            item = items[record['id']]
            self.assertEqual(item['library_tier'], 'task_example')
            self.assertEqual(item['rule'], 'red_light')
            self.assertEqual(item['image_size'], 1280)
            self.assertEqual(item['confidence'], .05)
        for withdrawn in ('curated-tu-v1', 'curated-tu-v9', 'curated-tu-v28', 'curated-tu-v5'):
            self.assertNotIn(withdrawn, items)
        self.assertTrue(all(item.get('library_tier') == 'supplementary'
                            for key, item in items.items() if key.startswith('curated-')))

    def test_catalogue_excludes_missing_unverified_and_outside_files(self):
        with tempfile.TemporaryDirectory(dir=web_ui.DATA_DIR) as temporary:
            root = Path(temporary)
            video = root/'video.mp4'; video.write_bytes(b'fixture')
            relative = video.relative_to(web_ui.PROJECT_ROOT).as_posix()
            legacy = root/'legacy.json'; curated = root/'curated.json'
            legacy.write_text(json.dumps({'candidates': [{'id': 'original', 'path': relative, 'rule': 'wrong_way', 'status': 'accepted_for_manual_review'}]}))
            def item(name, **extra):
                return dict({'id': name, 'path': relative, 'rule': 'wrong_way', 'status': 'accepted_for_manual_review',
                             'library_tier': 'curated', 'scene_type': 'intersection'}, **extra)
            videos = [item('verified', full_decode_verified=True), item('not-verified'),
                      item('roundabout', full_decode_verified=True, scene_type='roundabout'),
                      dict(item('retired', full_decode_verified=True), rule='failure_to_stop'),
                      item('original', full_decode_verified=True),
                      dict(item('outside', full_decode_verified=True), path='../outside.mp4'),
                      dict(item('missing', full_decode_verified=True), path='data/missing-curation-test.mp4')]
            curated.write_text(json.dumps({'videos': videos}))
            with patch.object(web_ui, 'CANDIDATE_CATALOG_PATH', legacy), patch.object(web_ui, 'CURATED_LIBRARY_PATH', curated):
                result = web_ui.load_evaluation_candidates()
            self.assertEqual([x['id'] for x in result], ['original', 'verified'])

    def test_screened_video_metadata_and_poster_are_available_without_exposing_other_files(self):
        with tempfile.TemporaryDirectory(dir=web_ui.DATA_DIR) as temporary:
            root = Path(temporary)
            video = root/'video.mp4'; video.write_bytes(b'video-fixture')
            poster = root/'video.jpg'; poster.write_bytes(b'poster-fixture')
            curated = root/'library.json'
            item = {'id': 'screened', 'path': video.relative_to(web_ui.PROJECT_ROOT).as_posix(),
                    'poster_path': poster.relative_to(web_ui.PROJECT_ROOT).as_posix(),
                    'status': 'accepted_for_manual_review', 'library_tier': 'curated',
                    'rule': 'wrong_way',
                    'scene_type': 'intersection',
                    'full_decode_verified': True, 'training_ready': False, 'evaluation_ready': False,
                    'source_id': 'one-physical-camera', 'role': 'control_candidate'}
            curated.write_text(json.dumps({'videos': [item]}))
            with patch.object(web_ui, 'CANDIDATE_CATALOG_PATH', root/'absent.json'), patch.object(web_ui, 'CURATED_LIBRARY_PATH', curated):
                server = ThreadingHTTPServer(('127.0.0.1', 0), web_ui.make_handler(web_ui.AppState()))
                worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
                try:
                    base = f'http://127.0.0.1:{server.server_port}'
                    with urlopen(base+'/api/dataset-candidates') as response:
                        record = json.load(response)['candidates'][0]
                    self.assertFalse(record['training_ready'])
                    self.assertFalse(record['evaluation_ready'])
                    self.assertEqual(record['source_id'], 'one-physical-camera')
                    with urlopen(base+record['poster_url']) as response:
                        self.assertEqual(response.read(), b'poster-fixture')
                        self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base+'/media/candidate-poster/unknown')
                    self.assertEqual(error.exception.code, 404)
                finally:
                    server.shutdown(); server.server_close(); worker.join()
