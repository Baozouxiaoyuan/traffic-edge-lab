from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from traffic_edge.web_ui import AppState


class JobQueueTests(unittest.TestCase):
    def test_only_one_detector_run_is_loaded_at_a_time(self):
        state = AppState()
        for name in ('first', 'second'):
            state.jobs[name] = {'id': name, 'status': 'queued'}
        first_started, release_first, second_requested = (threading.Event() for _ in range(3))
        starts = []

        def pipeline(video, *_args, **_kwargs):
            starts.append(video.name)
            if video.name == 'first':
                first_started.set()
                if not release_first.wait(5):
                    raise RuntimeError('Test did not release the first job')
            return {'candidate_events': 0}

        def run(name):
            if name == 'second':
                second_requested.set()
            state._run_job(name, Path(name), Path('unused'), Path('unused'), 'demo')

        with patch('traffic_edge.web_ui.run_pipeline', side_effect=pipeline):
            first = threading.Thread(target=run, args=('first',))
            second = threading.Thread(target=run, args=('second',))
            first.start()
            try:
                self.assertTrue(first_started.wait(2))
                second.start()
                self.assertTrue(second_requested.wait(2))
                self.assertEqual(state.get_job('second')['status'], 'queued')
                self.assertEqual(starts, ['first'])
            finally:
                release_first.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(starts, ['first', 'second'])
        self.assertEqual(state.get_job('second')['status'], 'complete')
