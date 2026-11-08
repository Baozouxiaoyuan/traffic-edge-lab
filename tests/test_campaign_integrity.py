import copy
import json
from pathlib import Path
import tempfile
import unittest

from traffic_edge.campaign import _campaign_inputs, _freeze_campaign_inputs, validate_campaign


class CampaignIntegrityTests(unittest.TestCase):
    def test_renaming_source_groups_does_not_hide_identical_split_videos(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            (base/'a.mp4').write_bytes(b'identical-source')
            (base/'b.mp4').write_bytes(b'identical-source')
            (base/'config.json').write_text('{}');(base/'truth.json').write_text('[]')
            (base/'model.pt').write_bytes(b'model')
            manifest={'variants':[{'name':str(i),'model':'model.pt','tracker':'bytetrack'} for i in range(2)],
                      'cases':[{'video_id':split,'source_group_id':split,'split':split,'status':'labelled',
                                'input_video':video,'config':'config.json','ground_truth':'truth.json'}
                               for split,video in [('validation','a.mp4'),('test','b.mp4')]]}
            path=base/'manifest.json';path.write_text(json.dumps(manifest))
            report=validate_campaign(path)
            self.assertFalse(report['ready'])
            self.assertTrue(any('Identical video content' in text for text in report['blockers']))

    def test_resume_refuses_changed_model_annotations_source_or_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            original={'model_hashes':{'baseline':'a'},'case_file_hashes':{'clip':{'config_sha256':'b'}},
                      'implementation_hashes':{'pipeline.py':'c'}}
            _freeze_campaign_inputs(output, original, False)
            _freeze_campaign_inputs(output, original, True)
            for key in original:
                changed=copy.deepcopy(original);changed[key]={'changed':'different'}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,'changed'):
                    _freeze_campaign_inputs(output, changed, True)
            self.assertEqual(json.loads((output/'campaign_inputs.json').read_text()),original)

    def test_old_unverified_runs_cannot_be_adopted_as_fresh_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);(output/'old-summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unverified'):
                _freeze_campaign_inputs(output, {}, True)

    def test_ledger_hashes_actual_weights_and_review_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            for name in ('source.mp4','config.json','truth.json','truth.review.json','model.pt'):
                (base/name).write_text('original')
            manifest={'cases':[{'video_id':'clip','input_video':'source.mp4','config':'config.json',
                               'ground_truth':'truth.json'}],
                      'variants':[{'name':'baseline','model':'model.pt'}]}
            path=base/'manifest.json';path.write_text(json.dumps(manifest))
            original=_campaign_inputs(path,manifest)
            for name in ('source.mp4','config.json','truth.json','truth.review.json','model.pt'):
                with self.subTest(file=name):
                    (base/name).write_text('changed')
                    self.assertNotEqual(_campaign_inputs(path,manifest),original)
                    (base/name).write_text('original')
