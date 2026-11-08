from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from traffic_edge import web_ui
from traffic_edge import collision_review
from traffic_edge.collision_review import combine_labels,scan_windows,validated_interval


class CollisionReviewTests(unittest.TestCase):
    def test_disagreement_and_missing_evidence_never_become_a_collision(self):
        for full in ('A','B','C','invalid'):
            for local in ('A','B','C','invalid'):
                result=combine_labels(full,local)
                self.assertEqual(result=='collision_review_candidate',full==local=='B')
        self.assertEqual(combine_labels('A','B'),'uncertain')
        self.assertEqual(combine_labels('B','B',False),'uncertain')
        self.assertEqual(combine_labels('A','A',False),'uncertain')

    def test_interval_validation_rejects_unbounded_or_outside_work(self):
        for start,end,duration in ((0,float('nan'),10),(0,float('inf'),10),(-1,4,10),
                                   (0,31,60),(0,1,10),(7,12,10)):
            with self.subTest(start=start,end=end):
                with self.assertRaises(ValueError):validated_interval(start,end,duration)
        self.assertEqual(validated_interval(2,8,10),(2,8))

    def test_uniform_scan_does_not_drop_the_unaligned_end_of_a_clip(self):
        windows=scan_windows(3,12.7)
        self.assertEqual(windows[0][0],3)
        self.assertAlmostEqual(windows[-1][1],12.7)
        self.assertTrue(all(a[1]>=b[0] for a,b in zip(windows,windows[1:])))
        self.assertEqual(scan_windows(0,2.5),[(0,2.5)])

    def test_scene_cut_only_invalidates_windows_touching_the_cut(self):
        flow={'camera_or_cut_rejections':1,'quality_intervals':[[6.2,6.4]]}
        self.assertTrue(collision_review.window_quality_ok(flow,0,4))
        self.assertFalse(collision_review.window_quality_ok(flow,4,8))
        self.assertFalse(collision_review.window_quality_ok(flow,6.3,10))
        self.assertTrue(collision_review.window_quality_ok(flow,8,12))

    def test_scene_review_assets_are_job_scoped_and_cannot_expose_other_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);(output/'window-000.jpg').write_bytes(b'review-image')
            (output/'worker.stderr.log').write_text('internal log')
            state=web_ui.AppState()
            state.jobs['scene']={'kind':'collision_review','output_dir':str(output)}
            state.jobs['rules']={'output_dir':str(output)}
            server=ThreadingHTTPServer(('127.0.0.1',0),web_ui.make_handler(state))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            try:
                base=f'http://127.0.0.1:{server.server_port}'
                with urlopen(base+'/api/collision-review-asset?id=scene&name=window-000.jpg') as response:
                    self.assertEqual(response.read(),b'review-image')
                for query in ('id=rules&name=window-000.jpg','id=scene&name=worker.stderr.log',
                              'id=scene&name=../window-000.jpg'):
                    with self.assertRaises(HTTPError) as error:urlopen(base+'/api/collision-review-asset?'+query)
                    self.assertEqual(error.exception.code,404)
                request=Request(base+'/api/review-collisions',data=json.dumps({'video_path':'../outside.mp4','end_seconds':4}).encode(),headers={'Content-Type':'application/json'})
                with self.assertRaises(HTTPError) as error:urlopen(request)
                self.assertEqual(error.exception.code,400)
            finally:
                server.shutdown();server.server_close();worker.join()

    def test_new_incident_examples_do_not_restore_the_rejected_wrong_way_video(self):
        self.assertNotIn('wrong-way-two-lane',web_ui.REAL_EXAMPLE_VIDEOS)
        self.assertTrue(all('njit_wrong_way_car' not in str(item[1]) for item in web_ui.COLLISION_EXAMPLES.values()))
        self.assertIn('id="collision-step"',web_ui.HTML)

    def test_camera_quality_failure_blocks_even_two_affirmative_model_views(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=web_ui.DATA_DIR) as temporary:
            root=Path(temporary);video=root/'source.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,96))
            for _ in range(25):writer.write(np.zeros((96,160,3),dtype=np.uint8))
            writer.release()
            fake=type('Reviewer',(),{'__init__':lambda self,device='cpu':None,
                                    'classify':lambda self,*args:{'label':'B','raw_response':'B'}})
            with patch.object(collision_review,'model_available',return_value=True), \
                 patch.object(collision_review,'propose',return_value={'windows':[],'camera_or_cut_rejections':1}), \
                 patch.object(collision_review,'collect_tracks',return_value=[]), \
                 patch.object(collision_review,'VideoActionReviewer',fake):
                result=collision_review.review_video(video,root/'output',0,2.4)
            self.assertEqual(result['candidate_window_count'],0)
            self.assertEqual(result['uncertain_window_count'],1)
            self.assertEqual(result['windows'][0]['quality_flags'],['possible_camera_motion_or_scene_cut'])


if __name__=='__main__':unittest.main()
