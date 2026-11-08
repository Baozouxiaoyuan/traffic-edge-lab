"""Rules included in the current fixed-camera project scope."""

ACTIVE_RULES = ('red_light', 'wrong_way', 'restricted_zone')
ACTIVE_SCENE_TYPES = ('straight_road', 'intersection')


def validate_active_rules(enabled: dict) -> None:
    unsupported = [name for name, active in enabled.items() if active and name not in ACTIVE_RULES]
    if unsupported:
        raise ValueError('Rule outside the current project scope: ' + ', '.join(unsupported)
                         + '. Supported rules: red-light, wrong-way and restricted-zone entry.')


def validate_config_scope(config: dict) -> None:
    validate_active_rules(config.get('enabled_rules', {}))
    if config.get('rule', {}).get('event_type') == 'candidate_failure_to_stop':
        validate_active_rules({'failure_to_stop': True})
    for lane in config.get('lanes', []):
        validate_config_scope(lane)
