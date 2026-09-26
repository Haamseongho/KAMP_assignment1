"""Training-free generation and validation of the legacy grouped split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import moldguard as mg
from research_runtime import digest, execution_record, write_json

FEATURES = ('Injection_Time Filling_Time Plasticizing_Time Cycle_Time Clamp_Close_Time '
            'Cushion_Position Plasticizing_Position Clamp_Open_Position Max_Injection_Speed '
            'Max_Screw_RPM Average_Screw_RPM Max_Injection_Pressure Max_Switch_Over_Pressure '
            'Max_Back_Pressure Average_Back_Pressure Barrel_Temperature_1 Barrel_Temperature_2 '
            'Barrel_Temperature_3 Barrel_Temperature_4 Barrel_Temperature_5 Barrel_Temperature_6 '
            'Hopper_Temperature Mold_Temperature_3 Mold_Temperature_4').split()
COLUMNS = ['machine', 'source_row_id', 'label_value', 'feature_group', 'partition', 'oof_validation_fold']


def load_inputs(data_dir):
    if data_dir is None or not Path(data_dir).is_dir():
        raise FileNotFoundError('Official CSV directory missing; pass --data-dir')
    data, features, provenance = mg.load_data(Path(data_dir), True)
    if features != FEATURES or set(data.machine) != {'cn7', 'rg3'}:
        raise ValueError('Official feature/machine schema mismatch')
    return mg.with_groups(data, features), features, provenance


def build_table(data):
    development, holdout = mg.split_groups(data)
    groups = mg.group_table(data).loc[development]
    mapping = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=mg.SEED).split(groups, groups.stratum)):
        mapping.update({g: fold for g in groups.index.to_numpy()[valid]})
    table = data[['machine', mg.ID_COL, mg.TARGET, 'feature_group']].rename(
        columns={mg.ID_COL: 'source_row_id', mg.TARGET: 'label_value'}).copy()
    table['partition'] = np.where(data.feature_group.isin(development), 'development', 'holdout')
    table['oof_validation_fold'] = data.feature_group.map(mapping).astype('Int64')
    return table


def validate_membership(data, table):
    if table.columns.tolist() != COLUMNS or table.duplicated(['machine', 'source_row_id']).any():
        raise ValueError('Invalid split schema or duplicate identifiers')
    if not table.partition.isin(['development', 'holdout']).all():
        raise ValueError('Unknown split partition')
    if table.groupby('feature_group').partition.nunique().max() != 1:
        raise ValueError('Feature group crosses partitions')
    dev = table.partition.eq('development')
    if (not table.loc[dev, 'oof_validation_fold'].isin(range(5)).all()
            or table.loc[~dev, 'oof_validation_fold'].notna().any()
            or table.loc[dev].groupby('feature_group').oof_validation_fold.nunique().max() != 1):
        raise ValueError('Invalid or overlapping group folds')
    expected = build_table(data)
    keys = ['machine', 'source_row_id']
    try:
        pd.testing.assert_frame_equal(table.sort_values(keys).reset_index(drop=True),
                                      expected.sort_values(keys).reset_index(drop=True), check_dtype=False)
    except AssertionError as error:
        raise ValueError('Split source IDs, labels, groups or frozen assignments differ') from error


def generate(data_dir, output, reference=None):
    data, features, provenance = load_inputs(data_dir)
    table = build_table(data)
    validate_membership(data, table)
    table.to_csv(Path(output) / 'split_manifest.csv', index=False)
    split = {'seed': mg.SEED, 'source_files': provenance, 'feature_columns': features,
             'group_rule': 'machine + all 24 original features; pandas row hash',
             'rows_by_partition': {str(k): int(v) for k, v in table.partition.value_counts().items()},
             'cross_partition_feature_groups': 0}
    result = {'schema_version': 'moldguard-split-v1', 'status': 'complete', 'split': split,
              'split_sha256': digest(Path(output) / 'split_manifest.csv'),
              'reference_comparison': {'status': 'not_provided_new_split'}}
    if reference is not None:
        load_validated(data_dir, reference)
        old = Path(reference) / 'split_manifest.csv'
        same = digest(old) == result['split_sha256']
        result['reference_comparison'] = {'status': 'identical' if same else 'different', 'sha256': digest(old)}
        if not same:
            raise ValueError('Generated split differs from supplied reference')
    write_json(Path(output) / 'run_manifest.json', result)
    return result


def load_validated(data_dir, split_dir):
    data, features, provenance = load_inputs(data_dir)
    split_dir = Path(split_dir)
    if not (split_dir / 'run_manifest.json').is_file() or not (split_dir / 'split_manifest.csv').is_file():
        raise FileNotFoundError('Split missing. Run research_split.py generate --data-dir ... --output-dir ... first')
    if (split_dir / 'execution.json').exists():
        if json.loads((split_dir / 'execution.json').read_text())['status'] != 'passed':
            raise ValueError('Split execution is not complete')
    record = json.loads((split_dir / 'run_manifest.json').read_text())
    if record.get('status') not in ('complete', 'passed') or record['split']['seed'] != mg.SEED:
        raise ValueError('Unfinished split or unsupported seed')
    for name, source in provenance.items():
        if source['sha256'] != record['split']['source_files'][name]['sha256']:
            raise ValueError('Input hash changed; a new reviewed split is required')
    expected_hash = record.get('split_sha256', record.get('output_hashes', {}).get('split_manifest.csv'))
    if expected_hash and expected_hash != digest(split_dir / 'split_manifest.csv'):
        raise ValueError('Split manifest hash mismatch')
    table = pd.read_csv(split_dir / 'split_manifest.csv')
    validate_membership(data, table)
    merged = data.merge(table.drop(columns=['label_value', 'feature_group']).rename(columns={'source_row_id': mg.ID_COL}),
                        on=['machine', mg.ID_COL], validate='one_to_one', sort=False)
    return merged, features, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'verify'])
    parser.add_argument('--data-dir', type=Path, default=mg.DATA_DIR)
    parser.add_argument('--output-dir', type=Path, help='New folder for generate; split folder for verify')
    parser.add_argument('--reference-split-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if args.command == 'generate':
        with execution_record(args.output_dir):
            print(json.dumps(generate(args.data_dir, args.output_dir, args.reference_split_dir), indent=2))
    else:
        data, _, _ = load_validated(args.data_dir, args.output_dir)
        print(json.dumps({'status': 'passed', 'rows': len(data), 'split_sha256': digest(args.output_dir / 'split_manifest.csv')}))


if __name__ == '__main__':
    main()
