"""Readiness checklist only: does not invent links, data, or ablation gains."""
import argparse
import json
from pathlib import Path


def assess(plan):
    if plan.get('competition_track') == 'general_student' and plan.get('track_evidence') and not plan.get('optional_fusion_requested', False):
        return {'status': 'not_required_for_confirmed_track', 'blockers': [],
                'ablation_run': False, 'claim_of_improvement_allowed': False,
                'note': 'General/student task 01 selects one dataset. Optional external data still needs provenance and validation.'}
    missing = []
    if plan.get('competition_track') not in ('general_student', 'employee') or not plan.get('track_evidence'):
        missing.append('competition_track_not_confirmed')
    primary = plan.get('primary') or {}
    auxiliary = plan.get('auxiliary') or {}
    if not auxiliary.get('dataset_id') or auxiliary.get('dataset_id') == primary.get('dataset_id'):
        missing.append('second_distinct_dataset_not_verified')
    for key in ('source_url', 'files_and_sha256', 'process', 'machines', 'variables_and_units', 'lineage', 'role'):
        if not auxiliary.get(key):
            missing.append('auxiliary_' + key + '_missing')
    linkage = plan.get('linkage', {})
    if linkage.get('method') not in ('validated_key_join', 'validated_time_window', 'feature_transfer', 'model_combination', 'decision_combination'):
        missing.append('linkage_method_missing')
    for key in ('pre_prediction_availability_evidence', 'scientific_rationale'):
        if not linkage.get(key):
            missing.append(key + '_missing')
    if linkage.get('method') in ('validated_key_join', 'validated_time_window'):
        for key in ('shared_physical_key_evidence', 'time_unit_alignment_evidence'):
            if not linkage.get(key):
                missing.append(key + '_missing')
    for key in ('physical_units_confirmed', 'label_semantics_confirmed', 'prediction_time_confirmed'):
        if primary.get(key) is not True:
            missing.append('primary_' + key + '_missing')
    return {'status': 'blocked' if missing else 'requires_independent_evidence_review',
            'blockers': missing, 'ablation_run': False, 'claim_of_improvement_allowed': False,
            'note': 'Metadata completeness alone is not data validation or competition compliance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.plan.read_text()))
    print(json.dumps(result, indent=2))
    if result['status'] == 'blocked':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
