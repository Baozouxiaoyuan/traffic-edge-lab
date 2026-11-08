from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traffic_edge.storage import atomic_write_json


class StorageTests(unittest.TestCase):
    def test_completed_replacement_is_valid_unicode_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'nested'/'draft.json'
            atomic_write_json(path, {'reviewer': '测试', 'labels': [1, 2]})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['reviewer'], '测试')
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_interrupted_replacement_preserves_previous_saved_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'draft.json'
            atomic_write_json(path, {'labels': ['original']})
            with patch('traffic_edge.storage.os.replace', side_effect=PermissionError('file locked')):
                with self.assertRaises(PermissionError):
                    atomic_write_json(path, {'labels': ['replacement']})
            self.assertEqual(json.loads(path.read_text())['labels'], ['original'])
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_nonfinite_payload_does_not_damage_previous_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'draft.json'
            atomic_write_json(path, {'duration': 10})
            with self.assertRaises(ValueError):
                atomic_write_json(path, {'duration': float('nan')})
            self.assertEqual(json.loads(path.read_text())['duration'], 10)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])
