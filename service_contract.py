"""Local reviewer CLI for versioned evidence; never grants operational approval."""
import argparse
import json
from pathlib import Path
import moldguard as mg
from moldguard_service.contracts import ContractRegistry, ITEMS, STATES, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['show', 'record'])
    parser.add_argument('--data-dir', type=Path, default=mg.DATA_DIR)
    parser.add_argument('--registry-dir', type=Path, default=mg.ROOT / 'service_state/contracts')
    parser.add_argument('--item', choices=ITEMS)
    parser.add_argument('--status', choices=STATES)
    parser.add_argument('--reviewer')
    parser.add_argument('--reason')
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    registry = ContractRegistry(args.registry_dir)
    sources = source_hashes(args.data_dir)
    if args.command == 'record':
        if not all((args.item, args.status, args.reviewer, args.reason)):
            parser.error('record requires --item --status --reviewer --reason')
        result = registry.record(args.item, args.status, sources, args.reviewer, args.reason, args.evidence)
    else:
        result = registry.view(sources)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
