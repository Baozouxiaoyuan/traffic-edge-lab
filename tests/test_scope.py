import json
from pathlib import Path
import tempfile
import unittest

from traffic_edge.pipeline import run_pipeline
from traffic_edge.campaign import validate_campaign
from traffic_edge.scope import ACTIVE_RULES, validate_active_rules
from traffic_edge.web_ui import create_config, validate_annotation_quality, HTML


class ActiveScopeTests(unittest.TestCase):
    def test_active_project_has_three_rules_and_keeps_red_light_stop_line_tool(self):
        self.assertEqual(set(ACTIVE_RULES), {'red_light', 'wrong_way', 'restricted_zone'})
        self.assertNotIn('id="noStopRule"', HTML)
        self.assertNotIn('value="candidate_failure_to_stop"', HTML)
        self.assertNotIn('/media/example/idiap-late-stop', HTML)
        self.assertIn('id="lineMode"', HTML)
        self.assertIn('3 rules', HTML)

    def test_retired_rule_cannot_be_enabled_by_a_direct_api_configuration(self):
        with self.assertRaisesRegex(ValueError, 'outside the current project scope'):
            create_config('test', [], .25, enabled_rules={'failure_to_stop': True})
        validate_active_rules({'red_light': True, 'failure_to_stop': False})

    def test_retired_event_cannot_be_saved_as_ground_truth(self):
        labels = [{'timestamp_seconds': 1.0, 'vehicle_bbox_normalized': [.2, .2, .4, .5],
                   'candidate_event': 'candidate_failure_to_stop', 'status': 'confirmed'}]
        issues = validate_annotation_quality(labels, 5.0)
        self.assertTrue(any(item['code'] == 'unknown_rule' and item['severity'] == 'error'
                            for item in issues))

    def test_old_cli_config_cannot_reactivate_the_retired_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root/'config.json'
            config.write_text(json.dumps({'enabled_rules': {'failure_to_stop': True}}))
            with self.assertRaisesRegex(ValueError, 'outside the current project scope'):
                run_pipeline(root/'unused.mp4', root/'run', config, 'demo')
            self.assertFalse((root/'run').exists())

    def test_campaign_dry_run_rejects_retired_rules_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'video.mp4').write_bytes(b'fixture')
            (root/'model.pt').write_bytes(b'fixture')
            (root/'truth.json').write_text('[]')
            (root/'config.json').write_text(json.dumps({'enabled_rules': {'failure_to_stop': True}}))
            manifest = {'variants': [dict(name=name, model='model.pt', tracker=name) for name in ('bytetrack', 'botsort')],
                        'cases': [dict(video_id=split, source_group_id=split, split=split, status='labelled',
                                       input_video='video.mp4', config='config.json', ground_truth='truth.json')
                                  for split in ('validation', 'test')]}
            path = root/'campaign.json'; path.write_text(json.dumps(manifest))
            result = validate_campaign(path)
            self.assertFalse(result['ready'])
            self.assertTrue(any('failure_to_stop' in blocker for blocker in result['blockers']))
