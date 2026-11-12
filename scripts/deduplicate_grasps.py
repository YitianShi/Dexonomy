#!/usr/bin/env python3
"""Remove near-identical saved poses without changing taxonomy labels or simulation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dexonomy.util.grasp_dedup import PoseThresholds, deduplicate, subset_rows
from scripts.filter_graspnet_scene_grasps import atomic_json, digest, output_lock

DATASETS = ('graspnet_single', 'shapenet_single', 'shapenet_clutter',
            'graspnet_test_clutter', 'graspnet_clutter')
COUNT_KEYS = ('input_rows', 'kept_rows', 'removed_rows', 'exact_pose_duplicates',
              'near_pose_duplicates', 'input_files', 'output_files', 'output_bytes')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=REPO_ROOT/'output/dataset_index.json')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--output', type=Path, default=REPO_ROOT/'work_runs/grasp_dedup_shadow')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--position-mm', type=float, default=2.)
    parser.add_argument('--rotation-deg', type=float, default=5.)
    parser.add_argument('--joint-rms-deg', type=float, default=3.)
    parser.add_argument('--joint-max-deg', type=float, default=8.)
    parser.add_argument('--group', default='*', help='Object/scale or scene glob')
    parser.add_argument('--max-groups', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_groups < 0:
        parser.error('workers must be positive; max-groups must be nonnegative')
    args.catalog, args.output = args.catalog.resolve(), args.output.resolve()
    args.thresholds = PoseThresholds(args.position_mm/1000, args.rotation_deg,
                                     args.joint_rms_deg, args.joint_max_deg)
    return args


def stat_token(path):
    stat = path.stat()
    return [stat.st_ino, stat.st_size, stat.st_mtime_ns]


def discover_jobs(args):
    catalog = json.loads(args.catalog.read_text())
    entries = {entry['id']: entry for entry in catalog['datasets']}
    jobs, sources = [], {}
    for name in args.datasets:
        entry = next((entries[key] for key in ('taxonomy_filtered_'+name,
                                               'deduplicated_'+name, name) if key in entries), None)
        if entry is None:
            raise ValueError(f'No saved grasp library for {name}')
        if entry['state'] != 'complete':
            raise ValueError(f'Source is incomplete: {entry["id"]}')
        root = (args.catalog.parent/entry['data']).resolve()
        if root.is_relative_to(args.output) or args.output.is_relative_to(root):
            raise ValueError('Deduplication output must be separate from every source library')
        single = entry['kind'] == 'single_object_grasp'
        groups = defaultdict(list)
        for taxonomy in sorted(root.iterdir()):
            if not taxonomy.is_dir():
                continue
            for path in sorted(taxonomy.glob('*/*/*.npy' if single else '*/*.npy')):
                relative = path.relative_to(taxonomy)
                group = relative.with_suffix('').as_posix() if single else relative.parent.name
                if fnmatch.fnmatch(group, args.group):
                    groups[group].append(path.relative_to(root).as_posix())
        keys = sorted(groups)
        if args.max_groups:
            keys = keys[:args.max_groups]
        sources[name] = {'id': entry['id'], 'root': str(root), 'kind': entry['kind'],
                         'expected_rows': entry['grasp_records'], 'expected_files': entry['npy_files'],
                         'groups': len(keys), 'coverage': entry.get('coverage', {})}
        jobs.extend({'dataset': name, 'key': key, 'root': str(root), 'files': groups[key]}
                    for key in keys)
    if not jobs:
        raise ValueError('No input files selected')
    return jobs, sources


def check_saved_subset(path, data, kept):
    saved = np.load(path, allow_pickle=True).item()
    expected = subset_rows(data, kept, saved['dedup_source_path'])
    if saved.keys() != expected.keys():
        raise AssertionError(f'Output fields differ: {path}')
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            # Diagnostic distance arrays may contain NaN or infinity.
            if value.dtype.kind in 'fc':
                np.testing.assert_array_equal(saved[key], value, err_msg=key)
            elif not np.array_equal(saved[key], value):
                raise AssertionError(f'Changed metadata {key}: {path}')
        elif saved[key] != value:
            raise AssertionError(f'Changed metadata {key}: {path}')


def process_group(job, run_root, thresholds):
    root = Path(job['root'])
    output_root = Path(run_root)/job['dataset']
    before = [stat_token(root/relative) for relative in job['files']]
    audit_path = output_root/'records'/f'{job["key"]}.npz'
    meta_path = audit_path.with_suffix('.json')
    signature = digest([job['files'], before])
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if old['signature'] == signature and audit_path.is_file():
            try:
                intact = all(stat_token(output_root/'filtered_grasps'/rel) == token
                             for rel, token in zip(old['files'], old['output_tokens']))
            except FileNotFoundError:
                intact = False
            if intact:
                return old
    counts = Counter()
    maxima = np.zeros(4)
    mappings, offsets, per_file = [], [0], []
    output_tokens = []
    started = time.monotonic()
    examples = []
    for relative in job['files']:
        path = root/relative
        data = np.load(path, allow_pickle=True).item()
        representative, result = deduplicate(data, thresholds)
        maxima = np.maximum(maxima, result.pop('max_removed_distances'))
        counts.update(result)
        kept = np.flatnonzero(representative == np.arange(len(representative)))
        if not len(kept):
            raise ValueError(f'Empty input file: {path}')
        destination = output_root/'filtered_grasps'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.npy.tmp')
        if result['removed_rows']:
            with temporary.open('wb') as stream:
                np.save(stream, subset_rows(data, kept, path))
            check_saved_subset(temporary, data, kept)
        else:
            shutil.copy2(path, temporary)
        temporary.replace(destination)
        token = stat_token(destination)
        output_tokens.append(token)
        counts.update(input_files=1, output_files=1, output_bytes=token[1])
        mappings.append(representative)
        offsets.append(offsets[-1]+len(representative))
        per_file.append([result[key] for key in COUNT_KEYS[:5]])
        if len(examples) < 3 and result['removed_rows']:
            removed = int(np.flatnonzero(representative != np.arange(len(representative)))[0])
            examples.append({'file': relative, 'removed_row': removed,
                             'kept_row': int(representative[removed])})
    if before != [stat_token(root/relative) for relative in job['files']]:
        raise ValueError(f'Input changed during deduplication: {job["key"]}')
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.with_suffix('.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, source_files=np.asarray(job['files']),
                            file_offsets=np.asarray(offsets, dtype=np.int64),
                            representative_indices=np.concatenate(mappings),
                            per_file_counts=np.asarray(per_file, dtype=np.int64),
                            count_names=np.asarray(COUNT_KEYS[:5]))
    audit_path.with_suffix('.npz.tmp').replace(audit_path)
    metadata = {'dataset': job['dataset'], 'group': job['key'], 'signature': signature,
                'source_root': str(root), 'input_tokens': before, 'files': job['files'],
                'output_tokens': output_tokens, 'counts': dict(counts),
                'max_removed_distances': maxima.tolist(), 'examples': examples,
                'seconds': time.monotonic()-started,
                'verification': 'Every removed pose checked against a retained representative in all phases; '
                                'all fields of every changed output reloaded and compared to source subset'}
    atomic_json(meta_path, metadata)
    return metadata


def register_datasets(args, run_root, sources, totals):
    catalog = json.loads(args.catalog.read_text())
    entries = {entry['id']: entry for entry in catalog['datasets']}
    for name, spec in sources.items():
        source = entries[spec['id']]
        result = totals[name]
        entry = {
            'id': 'deduplicated_'+name, 'kind': source['kind'], 'state': 'complete',
            'root': os.path.relpath(run_root/name, args.catalog.parent),
            'data': os.path.relpath(run_root/name/'filtered_grasps', args.catalog.parent),
            'progress': os.path.relpath(run_root/'progress.json', args.catalog.parent),
            'source_dataset': spec['id'], 'grasp_records': result['kept_rows'],
            'removed_duplicates': result['removed_rows'], 'npy_files': result['output_files'],
            'logical_bytes': result['output_bytes'], 'coverage': spec['coverage'],
            'deduplication_run': run_root.name, 'deduplication_thresholds': asdict(args.thresholds),
            'deduplication_scope': 'same assigned taxonomy, object/scale or scene/target; all three phases; '
                                   'identical target contact body masks',
            'labels_preserved': True,
        }
        if source['kind'] == 'clutter_grasp_instance':
            entry['scene_lift_validated'] = source.get('scene_lift_validated', False)
        catalog['datasets'] = [e for e in catalog['datasets'] if e['id'] != entry['id']]+[entry]
    catalog['recommended_datasets'] = ['deduplicated_'+name for name in sources]
    catalog['snapshot_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.catalog, catalog)


def run(args, jobs, sources):
    implementation = hashlib.sha256()
    for path in (Path(__file__), REPO_ROOT/'dexonomy/util/grasp_dedup.py'):
        implementation.update(path.read_bytes())
    settings = {'thresholds': asdict(args.thresholds), 'sources': sources,
                'group': args.group, 'max_groups': args.max_groups,
                'algorithm': 'first-row greedy direct representative; all phases; same contact body masks',
                'implementation_sha256': implementation.hexdigest()}
    run_root = args.output/'runs'/digest(settings)[:16]
    with output_lock(args.output):
        run_root.mkdir(parents=True, exist_ok=True)
        atomic_json(run_root/'settings.json', settings)
        totals = {name: Counter() for name in sources}
        completed = Counter()
        failed = 0
        started = time.monotonic()
        last_report = started
        def progress(state):
            result = {'state': state, 'pid': os.getpid(), 'run': str(run_root),
                      'updated_at': datetime.now(timezone.utc).isoformat(),
                      'elapsed_seconds': time.monotonic()-started, 'workers': args.workers,
                      'completed_groups': sum(completed.values()), 'total_groups': len(jobs),
                      'failed_groups': failed, 'thresholds': asdict(args.thresholds),
                      'datasets': {name: {'counts': dict(totals[name]),
                                          'completed_groups': completed[name], 'total_groups': spec['groups']}
                                   for name, spec in sources.items()}}
            atomic_json(run_root/'progress.json', result)
            atomic_json(args.output/'progress.json', result)
            return result
        for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[variable] = '1'
        executor = ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                       mp_context=multiprocessing.get_context('spawn'))
        pending = {}
        iterator = iter(jobs)
        exhausted = False
        try:
            progress('running')
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers*2:
                    try:
                        job = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[executor.submit(process_group, job, str(run_root), args.thresholds)] = job
                if not pending:
                    continue
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        record = future.result()
                        totals[job['dataset']].update(record['counts'])
                        completed[job['dataset']] += 1
                    except Exception as exc:
                        failed += 1
                        with (run_root/'errors.jsonl').open('a') as stream:
                            stream.write(json.dumps({'dataset': job['dataset'], 'group': job['key'],
                                                     'error': f'{type(exc).__name__}: {exc}'})+'\n')
                        print(f'ERROR {job["dataset"]}/{job["key"]}: {exc}', flush=True)
                if time.monotonic()-last_report >= 10:
                    result = progress('running')
                    print(json.dumps(result), flush=True)
                    last_report = time.monotonic()
            full = args.group == '*' and not args.max_groups
            if full:
                for name, spec in sources.items():
                    if (totals[name]['input_rows'] != spec['expected_rows']
                            or totals[name]['input_files'] != spec['expected_files']):
                        failed += 1
                        print(f'ERROR input count mismatch: {name}', flush=True)
            if not failed and full and args.output.is_relative_to(args.catalog.parent):
                register_datasets(args, run_root, sources, totals)
            result = progress('complete' if not failed else 'incomplete')
            print(json.dumps(result, indent=2), flush=True)
            return 0 if not failed else 1
        except BaseException:
            progress('interrupted')
            for process in list(executor._processes.values()):
                process.terminate()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    args = parse_args(argv)
    jobs, sources = discover_jobs(args)
    print(json.dumps({'jobs': len(jobs), 'sources': sources, 'thresholds': asdict(args.thresholds)}), flush=True)
    if args.dry_run:
        return 0
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    return run(args, jobs, sources)


if __name__ == '__main__':
    raise SystemExit(main())
