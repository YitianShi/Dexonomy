#!/usr/bin/env python3
"""Check required target contacts, optionally reject extra contact bodies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dexonomy.util.taxonomy_contacts import AMBIGUOUS, INVALID_POSE, NO_TAXONOMY, RETAINED, STATUS_NAMES, ContactReplay, ContactRules
from dexonomy.util.file_util import load_scene_cfg
from scripts.filter_graspnet_scene_grasps import atomic_json, digest, file_token, output_lock

DATASETS = ('graspnet_single', 'shapenet_single', 'shapenet_clutter', 'graspnet_test_clutter', 'graspnet_clutter')
PHASES = ('pregrasp_qpos', 'grasp_qpos', 'squeeze_qpos')
_WORKER_RULES = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=REPO_ROOT/'output/dataset_index.json')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--output', type=Path, default=REPO_ROOT/'work_runs/taxonomy_contact_filter_shadow')
    parser.add_argument('--workers', type=int, default=84)
    parser.add_argument('--contact-distance', type=float, default=.002)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--allow-extra-contacts', dest='allow_extra_contacts', action='store_true',
                        help='Restore rejected/ambiguous originals with required groups covered; preserve assigned labels (default)')
    policy.add_argument('--strict-extra-contacts', dest='allow_extra_contacts', action='store_false',
                        help='Reject any target contact outside the original taxonomy allowed bodies')
    parser.set_defaults(allow_extra_contacts=True)
    parser.add_argument('--reuse-contact-run', type=Path,
                        help='Reclassify a completed audit run without repeating MuJoCo contact queries')
    parser.add_argument('--group', default='*', help='Object/scale or scene name glob; default all')
    parser.add_argument('--max-groups', type=int, default=0, help='Diagnostic limit; 0 processes all')
    parser.add_argument('--max-per-file', type=int, default=0, help='Diagnostic limit; 0 processes all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.contact_distance <= 0 or args.max_groups < 0 or args.max_per_file < 0:
        parser.error('workers/distance must be positive; diagnostic limits nonnegative')
    args.output = args.output.resolve()
    args.catalog = args.catalog.resolve()
    if args.reuse_contact_run:
        args.reuse_contact_run = args.reuse_contact_run.resolve()
        if args.max_per_file:
            parser.error('--reuse-contact-run requires --max-per-file 0')
    return args


def discover_jobs(args, rules):
    entries = {entry['id']:entry for entry in json.loads(args.catalog.read_text())['datasets']}
    jobs, specifications = [], {}
    for dataset_id in args.datasets:
        entry = next((entries[key] for key in (dataset_id, 'deduplicated_'+dataset_id)
                      if key in entries), None)
        if entry is None:
            raise ValueError(f'No saved grasp library for {dataset_id}')
        root = (args.catalog.parent/entry['data']).resolve()
        if not root.is_dir():raise FileNotFoundError(root)
        if root.is_relative_to(args.output) or args.output.is_relative_to(root):
            raise ValueError('Output must be separate from all input libraries')
        single = entry['kind'] == 'single_object_grasp'
        groups = defaultdict(list)
        for taxonomy, name in enumerate(rules.names):
            pattern = '*/*/*.npy' if single else '*/*.npy'
            for path in sorted((root/name).glob(pattern)):
                relative = path.relative_to(root/name)
                key = relative.with_suffix('').as_posix() if single else relative.parent.name
                if fnmatch.fnmatch(key, args.group):
                    groups[key].append((taxonomy, path.relative_to(root).as_posix()))
        for key in sorted(groups):
            jobs.append({'dataset':dataset_id, 'key':key, 'root':str(root), 'single':single, 'files':groups[key]})
        specifications[dataset_id] = {'root':str(root), 'kind':entry['kind'],
                                      'expected_input_rows':entry.get('grasp_records'),
                                      'expected_input_files':entry.get('npy_files'), 'groups':len(groups)}
    if args.max_groups:jobs = jobs[:args.max_groups]
    if not jobs:raise ValueError('No input groups matched')
    return jobs, specifications


def worker_init():
    global _WORKER_RULES
    _WORKER_RULES = ContactRules.load(REPO_ROOT/'assets/hand/shadow/init_tmpl')


def input_signature(job):
    return digest([file_token(Path(job['root'])/relative) for _,relative in job['files']])


def save_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:np.save(stream, value)


def apply_contact_policy(audit, taxonomy, allow_extra_contacts):
    """Restore eligible excluded rows without changing any assigned labels."""
    strict_status = np.asarray(audit.get('strict_status', audit['status'])).copy()
    strict_assigned = np.asarray(audit.get('strict_assigned_taxonomy_ids', audit['assigned_taxonomy_ids'])).copy()
    strict_matches = np.asarray(audit.get('strict_matching_taxonomy_masks', audit['matching_taxonomy_masks'])).copy()
    audit.update(strict_status=strict_status, strict_assigned_taxonomy_ids=strict_assigned,
                 strict_matching_taxonomy_masks=strict_matches)
    audit['status'] = strict_status.copy()
    audit['assigned_taxonomy_ids'] = strict_assigned.copy()
    audit['matching_taxonomy_masks'] = strict_matches.copy()
    accepted = np.zeros(len(strict_status), dtype=bool)
    if allow_extra_contacts:
        recoverable = (strict_status == AMBIGUOUS) | (strict_status == NO_TAXONOMY)
        accepted = ((audit['missing_required_group_masks'] == 0)
                    & (audit['contact_body_masks'] != 0) & recoverable)
        audit['status'][accepted] = RETAINED
        audit['assigned_taxonomy_ids'][accepted] = taxonomy
        audit['matching_taxonomy_masks'][accepted] |= np.uint64(1 << taxonomy)
    audit['original_label_restored'] = accepted
    return {f'restored_from_{name}':int(np.count_nonzero(accepted & (strict_status == index)))
            for index,name in enumerate(STATUS_NAMES)}


def cached_contact_audit(job, settings, signature):
    """Validate historical source identities before reinterpreting their contacts."""
    base = Path(settings['reuse_contact_run'])
    path = base/job['dataset']/'records'/f"{job['key']}.npz"
    token = file_token(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if (metadata['input_signature'] != signature
                or Path(metadata['source_root']).resolve() != Path(job['root']).resolve()
                or metadata['dataset'] != job['dataset'] or metadata['group'] != job['key']):
            raise ValueError(f'Historical audit does not match current input: {path}')
        if archive['source_files'].tolist() != [relative for _,relative in job['files']]:
            raise ValueError(f'Historical audit file order differs: {path}')
        if archive['source_taxonomy_ids'].tolist() != [taxonomy for taxonomy,_ in job['files']]:
            raise ValueError(f'Historical taxonomy order differs: {path}')
        if any(file_token(item[0]) != item for item in metadata['scene_tokens']):
            raise ValueError(f'Historical scene configuration changed: {path}')
        offsets = archive['file_offsets'].copy()
        fields = ('status','assigned_taxonomy_ids','matching_taxonomy_masks','contact_body_masks',
                  'missing_required_group_masks','extra_contact_body_masks',
                  'source_template_max_point_distance_m','target_contact_counts',
                  'strict_status','strict_assigned_taxonomy_ids','strict_matching_taxonomy_masks')
        arrays = {key:archive[key].copy() for key in fields if key in archive}
        if (len(offsets) != len(job['files'])+1 or offsets[0] != 0
                or np.any(np.diff(offsets) < 0)
                or any(len(array) != offsets[-1] for array in arrays.values())):
            raise ValueError(f'Invalid historical audit row ranges: {path}')
    if file_token(path) != token:raise ValueError(f'Historical audit changed during read: {path}')
    return arrays, offsets, token


def subset_output(parts, destination, rules, label_status, original_label_policy=None):
    """Immediate source references preserve every original record's provenance."""
    count = sum(len(indices) for data,path,tax,indices,audit in parts)
    output = {phase:np.concatenate([np.asarray(data[phase], dtype=np.float32)[indices]
                                   for data,path,tax,indices,audit in parts]) for phase in PHASES}
    paths = np.empty(count, dtype=object)
    indices_out, originals, matches, bodies, point_distances = [], [], [], [], []
    anchors, offsets = [], 0
    for data,path,tax,indices,audit in parts:
        paths[offsets:offsets+len(indices)] = str(path)
        offsets += len(indices)
        indices_out.append(indices.astype(np.int32))
        originals.append(np.full(len(indices), tax, dtype=np.uint8))
        matches.append(audit['matching_taxonomy_masks'][indices])
        bodies.append(audit['contact_body_masks'][indices])
        point_distances.append(audit['source_template_max_point_distance_m'][indices])
        anchors.append(np.asarray(data.get('hand_contact_indices', np.full(len(data['grasp_qpos']),-1)), dtype=np.int32)[indices])
    first = parts[0][0]
    output.update(scene_path=str(first['scene_path']), target_name=str(first.get('target_name', parts[0][1].stem)),
                  source_grasp_paths=paths, source_indices=np.concatenate(indices_out),
                  original_taxonomy_ids=np.concatenate(originals),
                  matching_taxonomy_masks=np.concatenate(matches), contact_body_masks=np.concatenate(bodies),
                  source_template_max_point_distance_m=np.concatenate(point_distances),
                  source_hand_contact_indices=np.concatenate(anchors),
                  taxonomy_names=rules.names, taxonomy_contact_body_names=rules.bodies,
                  taxonomy_label_status=label_status, verified_by='saved_grasp_target_contact_taxonomy_v1')
    for field in ('scene_scale', 'effective_object_scale', 'relative_scale_error'):
        if all(field in data for data,path,tax,indices,audit in parts):
            output[field] = np.concatenate([np.asarray(data[field])[indices] for data,path,tax,indices,audit in parts])
    for field in ('strict_matching_taxonomy_masks', 'original_label_restored',
                  'extra_contact_body_masks', 'missing_required_group_masks'):
        if all(field in audit for data,path,tax,indices,audit in parts):
            output[field] = np.concatenate([audit[field][indices] for data,path,tax,indices,audit in parts])
    if 'original_label_restored' in output:
        output['verified_by'] = 'saved_grasp_target_contact_taxonomy_v2'
        output['taxonomy_original_label_policy'] = original_label_policy
    save_npy(destination, output)


def process_group(job, run_root, settings):
    rules = _WORKER_RULES
    started = time.monotonic()
    root = Path(job['root'])
    before = input_signature(job)
    stage = Path(tempfile.mkdtemp(prefix='job_', dir=Path(run_root)/'.staging'))
    sinks = defaultdict(list)
    audit_parts, offsets, targets, scene_paths = [], [0], [], []
    counts, transitions = Counter(), Counter()
    scene_tokens, scene_cache = {}, {}
    try:
        first_data = np.load(root/job['files'][0][1], allow_pickle=True).item()
        scene = load_scene_cfg(str(first_data['scene_path']))
        cached = cached_contact_audit(job, settings, before) if settings.get('reuse_contact_run') else None
        probe = None if cached else ContactReplay(scene, REPO_ROOT/'assets/hand/shadow/right.xml', rules, settings['contact_distance'])
        for file_number, (taxonomy, relative) in enumerate(job['files']):
            path = root/relative
            data = first_data if file_number == 0 else np.load(path, allow_pickle=True).item()
            poses = np.asarray(data['grasp_qpos'])
            if poses.ndim != 2 or poses.shape[1] != 29 or any(np.asarray(data[phase]).shape != poses.shape for phase in PHASES):
                raise ValueError(f'Mismatched pose arrays: {path}')
            count = min(len(poses), settings['max_per_file'] or len(poses))
            target = str(data.get('target_name', scene['task']['obj_name']))
            scene_key = str(data['scene_path'])
            if scene_key not in scene_cache:
                scene_tokens[scene_key] = file_token(scene_key)
                scene_cache[scene_key] = load_scene_cfg(scene_key)
            file_scene = scene_cache[scene_key]
            # A group may contain different targets, but must share the exact layout.
            if set(file_scene['scene']) != set(scene['scene']):raise ValueError(f'Layout mismatch: {path}')
            for name, cfg in scene['scene'].items():
                other = file_scene['scene'][name]
                if cfg.get('type') != other.get('type') or any(
                    key in cfg and not np.array_equal(cfg[key], other.get(key)) for key in ('pose','scale','xml_path')):
                    raise ValueError(f'Inconsistent scene geometry: {path}')
            if probe:probe.set_target(target)
            audit = {
                'status':np.zeros(count, np.uint8), 'assigned_taxonomy_ids':np.full(count,-1,np.int8),
                'matching_taxonomy_masks':np.zeros(count,np.uint64), 'contact_body_masks':np.zeros(count,np.uint64),
                'missing_required_group_masks':np.zeros(count,np.uint16), 'extra_contact_body_masks':np.zeros(count,np.uint64),
                'source_template_max_point_distance_m':np.full(count,np.nan,np.float32), 'target_contact_counts':np.zeros(count,np.uint16),
            }
            fields = {'status':'status','assigned_taxonomy_ids':'assigned','matching_taxonomy_masks':'matches',
                      'contact_body_masks':'bodies','missing_required_group_masks':'missing','extra_contact_body_masks':'extra',
                      'source_template_max_point_distance_m':'max_point_distance','target_contact_counts':'contact_count'}
            if cached:
                arrays, cached_offsets, _ = cached
                begin,end = map(int,cached_offsets[file_number:file_number+2])
                if end-begin != count:raise ValueError(f'Historical row count differs: {path}')
                audit = {name:array[begin:end].copy() for name,array in arrays.items()}
                covered = np.zeros(count,dtype=np.uint16)
                for group_index,group in enumerate(rules.required[taxonomy]):
                    covered |= ((audit['contact_body_masks'] & np.uint64(group)) == 0).astype(np.uint16) << group_index
                valid = audit.get('strict_status',audit['status']) != INVALID_POSE
                if not np.array_equal(covered[valid],audit['missing_required_group_masks'][valid]):
                    raise ValueError(f'Historical required groups differ: {path}')
            else:
                for index in range(count):
                    result = probe.query(poses[index], taxonomy)
                    for name,key in fields.items():audit[name][index] = result[key]
            counts.update(apply_contact_policy(audit, taxonomy, settings['allow_extra_contacts']))
            status_counts = np.bincount(audit['status'], minlength=len(STATUS_NAMES))
            counts.update({name:int(value) for name,value in zip(STATUS_NAMES,status_counts)})
            counts['input_rows'] += count
            counts['input_files'] += 1
            target_relative = Path(relative).relative_to(rules.names[taxonomy])
            for assigned in np.unique(audit['assigned_taxonomy_ids']):
                if assigned < 0:continue
                selected = np.flatnonzero(audit['assigned_taxonomy_ids'] == assigned)
                destination = Path('filtered_grasps')/rules.names[int(assigned)]/target_relative
                sinks[destination].append((data,path,taxonomy,selected,audit))
                transitions[f'{rules.names[taxonomy]}->{rules.names[int(assigned)]}'] += len(selected)
            ambiguous = np.flatnonzero(audit['status'] == AMBIGUOUS)
            if len(ambiguous):
                destination = Path('ambiguous_grasps')/rules.names[taxonomy]/target_relative
                sinks[destination].append((data,path,taxonomy,ambiguous,audit))
            audit_parts.append(audit)
            offsets.append(offsets[-1]+count)
            targets.append(target)
            scene_paths.append(str(data['scene_path']))
        if input_signature(job) != before:raise ValueError('Source records changed during replay')
        if any(file_token(path) != token for path,token in scene_tokens.items()):
            raise ValueError('Scene configuration changed during replay')
        for relative,parts in sinks.items():
            subset_output(parts, stage/relative, rules, 'ambiguous' if relative.parts[0]=='ambiguous_grasps' else 'assigned',
                          settings['original_label_policy'])
        counts['filtered_files'] = sum(path.parts[0] == 'filtered_grasps' for path in sinks)
        counts['ambiguous_files'] = len(sinks)-counts['filtered_files']
        counts['filtered_bytes'] = sum((stage/path).stat().st_size for path in sinks if path.parts[0]=='filtered_grasps')
        metadata = {'format':'taxonomy_contact_replay_v1', 'dataset':job['dataset'],'group':job['key'],
                    'source_root':str(root),'input_signature':before,'counts':dict(counts),
                    'transitions':dict(transitions),'scene_tokens':list(scene_tokens.values()),
                    'original_label_policy':settings['original_label_policy'],
                    'seconds':round(time.monotonic()-started,3)}
        if cached:metadata['source_audit_token'] = cached[2]
        arrays = {name:np.concatenate([part[name] for part in audit_parts]) for name in audit_parts[0]}
        with (stage/'audit.npz').open('wb') as stream:
            np.savez_compressed(stream, **arrays, file_offsets=np.asarray(offsets,np.int64),
                                source_files=np.asarray([relative for _,relative in job['files']]),
                                source_taxonomy_ids=np.asarray([tax for tax,_ in job['files']],np.uint8),
                                target_names=np.asarray(targets), scene_paths=np.asarray(scene_paths),
                                metadata=np.asarray(json.dumps(metadata)))
        return {'stage':str(stage),'signature':before,'metadata':metadata,
                'outputs':[str(relative) for relative in sinks]}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def publish(job, result, run_root):
    stage = Path(result['stage'])
    try:
        if input_signature(job) != result['signature']:raise ValueError('Input changed before publication')
        if any(file_token(token[0]) != token for token in result['metadata']['scene_tokens']):
            raise ValueError('Scene configuration changed before publication')
        audit_token = result['metadata'].get('source_audit_token')
        if audit_token and file_token(audit_token[0]) != audit_token:
            raise ValueError('Historical contact audit changed before publication')
        base = run_root/job['dataset']
        wanted = set(result['outputs'])
        for category in ('filtered_grasps','ambiguous_grasps'):
            for taxonomy in {taxonomy for taxonomy,_ in job['files']}:
                name = _WORKER_RULES.names[taxonomy]
                path = base/category/name/job['key']
                existing = [path.with_suffix('.npy')] if job['single'] else list(path.glob('*.npy'))
                for old in existing:
                    if old.is_file() and str(old.relative_to(base)) not in wanted:old.unlink()
        for relative in result['outputs']:
            final = base/relative;final.parent.mkdir(parents=True, exist_ok=True)
            (stage/relative).replace(final)
        final_audit = base/'records'/f"{job['key']}.npz"
        final_audit.parent.mkdir(parents=True, exist_ok=True)
        (stage/'audit.npz').replace(final_audit)
        return [file_token(base/relative) for relative in result['outputs']]+[file_token(final_audit)]
    finally:shutil.rmtree(stage, ignore_errors=True)


def implementation_signature():
    import mujoco
    files = [Path(__file__),REPO_ROOT/'dexonomy/util/taxonomy_contacts.py',
             REPO_ROOT/'dexonomy/sim/mujoco_env.py',REPO_ROOT/'dexonomy/sim/basic.py',
             REPO_ROOT/'dexonomy/util/file_util.py',REPO_ROOT/'dexonomy/util/mujoco_util.py']
    code = [[str(path),hashlib.sha256(path.read_bytes()).hexdigest()] for path in files]
    roots = [REPO_ROOT/'assets/hand/shadow',REPO_ROOT/'third_party/mujoco_menagerie/shadow_hand/assets',
             REPO_ROOT/'assets/object/graspnet_1billion/processed_data',
             REPO_ROOT/'assets/object/dexgraspnet2_shapenet/processed_data']
    assets = [file_token(path) for root in roots for path in sorted(root.rglob('*')) if path.is_file()]
    return digest([code,assets,mujoco.__version__,np.__version__])


def register_filtered_dataset(catalog_path, source_id, run_root, counts, run_id, allow_extra_contacts=False):
    catalog = json.loads(catalog_path.read_text())
    source = next(entry for entry in catalog['datasets']
                  if entry['id'] in (source_id, 'deduplicated_'+source_id))
    entry = {
        'id':'taxonomy_filtered_'+source_id, 'kind':source['kind'], 'state':'complete',
        'root':str((run_root/source_id).relative_to(catalog_path.parent)),
        'data':str((run_root/source_id/'filtered_grasps').relative_to(catalog_path.parent)),
        'progress':str((run_root/'progress.json').relative_to(catalog_path.parent)),
        'grasp_records':counts['retained']+counts['relabeled'],
        'npy_files':counts['filtered_files'], 'logical_bytes':counts['filtered_bytes'],
        'source_dataset':source['id'],'contact_filter_run':run_id,
        'taxonomy_contact_counts':dict(counts),
        'ambiguous_data':str((run_root/source_id/'ambiguous_grasps').relative_to(catalog_path.parent)),
        'label_scope':('preserve assigned labels; restore excluded originals with required target-contact groups covered; extra bodies allowed'
                       if allow_extra_contacts else 'strict target-contact body rules; does not infer full grasp posture taxonomy'),
        'allow_extra_contacts':allow_extra_contacts,
    }
    catalog['datasets'] = [current for current in catalog['datasets'] if current['id'] != entry['id']]+[entry]
    catalog['snapshot_at'] = datetime.now(ZoneInfo(catalog.get('timezone','Europe/Berlin'))).isoformat()
    atomic_json(catalog_path,catalog)


def run(args, jobs, specifications, rules):
    import mujoco
    settings = {'datasets':args.datasets,'sources':specifications,'group':args.group,
                'max_groups':args.max_groups,'max_per_file':args.max_per_file,
                'contact_distance':args.contact_distance,'pose':'grasp_qpos',
                'strict_extra_contacts':not args.allow_extra_contacts,
                'allow_extra_contacts':args.allow_extra_contacts,
                'original_label_policy':('preserve_assigned_restore_excluded_required_groups' if args.allow_extra_contacts else 'strict'),
                'position_distance_is_diagnostic_only':True,
                'mujoco_version':mujoco.__version__,'rules':rules.manifest(),
                'implementation':implementation_signature()}
    if args.reuse_contact_run:
        source_settings_path = args.reuse_contact_run/'settings.json'
        source_settings = json.loads(source_settings_path.read_text())
        source_progress = json.loads((args.reuse_contact_run/'progress.json').read_text())
        if source_progress.get('state') != 'complete':raise ValueError('Contact audit run is not complete')
        if (source_settings['contact_distance'] != args.contact_distance or source_settings['pose'] != 'grasp_qpos'
                or source_settings['rules'] != rules.manifest()):
            raise ValueError('Contact audit distance, phase or taxonomy definitions differ')
        for dataset in args.datasets:
            if source_settings['sources'].get(dataset,{}).get('root') != specifications[dataset]['root']:
                raise ValueError(f'Contact audit uses a different source library: {dataset}')
        settings.update(reuse_contact_run=str(args.reuse_contact_run),
                        source_settings_token=file_token(source_settings_path),
                        contact_query_engine=source_settings['mujoco_version'])
    run_id = digest(settings)[:16]
    run_root = args.output/'runs'/run_id
    with output_lock(args.output):
        staging = run_root/'.staging';staging.mkdir(parents=True,exist_ok=True)
        for path in staging.iterdir():
            if path.is_dir():shutil.rmtree(path)
        atomic_json(run_root/'settings.json',settings)
        db = sqlite3.connect(run_root/'progress.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS completed (job_key TEXT PRIMARY KEY,signature TEXT,record TEXT)')
        counts, dataset_counts, transitions = Counter(), defaultdict(Counter), Counter()
        finished = resumed = failed = 0
        expected_jobs = Counter(job['dataset'] for job in jobs)
        completed_jobs = Counter()
        started = time.monotonic();last_report = 0
        fresh_rows = 0
        registered = set()
        iterator = iter(jobs);pending = {};exhausted = False
        full_selection = not args.max_groups and not args.max_per_file and args.group == '*'
        def account(job, metadata, fresh=True):
            nonlocal fresh_rows
            counts.update(metadata['counts']);dataset_counts[job['dataset']].update(metadata['counts'])
            transitions.update(metadata['transitions']);completed_jobs[job['dataset']] += 1
            if fresh:fresh_rows += metadata['counts']['input_rows']
            key = job['dataset']
            spec = specifications[key]
            if (full_selection and args.output.is_relative_to(args.catalog.parent)
                    and completed_jobs[key] == expected_jobs[key]
                    and dataset_counts[key]['input_rows'] == spec['expected_input_rows']
                    and dataset_counts[key]['input_files'] == spec['expected_input_files']
                    and key not in registered):
                register_filtered_dataset(args.catalog,key,run_root,dataset_counts[key],run_id,args.allow_extra_contacts)
                registered.add(key)
        def progress(state):
            nonlocal last_report
            elapsed = time.monotonic()-started
            estimate_total = sum(spec['expected_input_rows'] or 0 for spec in specifications.values()) if full_selection else None
            result = {'state':state,'pid':os.getpid(),'run_id':run_id,'run_root':str(run_root),
                      'updated_at':datetime.now(timezone.utc).isoformat(),'workers':args.workers,'gpu':False,
                      'jobs_total':len(jobs),'jobs_completed':finished,'jobs_resumed':resumed,'jobs_failed':failed,
                      'active_jobs':len(pending),'counts':dict(counts),'expected_input_rows':estimate_total,
                      'rows_per_second':round(fresh_rows/elapsed,1) if elapsed else 0,
                      'elapsed_seconds':round(elapsed,1),
                      'rules':{'distance_m':args.contact_distance,'phase':'grasp','strict':not args.allow_extra_contacts,
                               'preserve_assigned_labels':True},
                      'datasets':{key:{'completed_groups':completed_jobs[key],'total_groups':expected_jobs[key],
                                       'counts':dict(dataset_counts[key]),
                                       'filtered_grasps':str(run_root/key/'filtered_grasps'),
                                       'ambiguous_grasps':str(run_root/key/'ambiguous_grasps')}
                                  for key in args.datasets}}
            atomic_json(args.output/'progress.json',result);atomic_json(run_root/'progress.json',result)
            atomic_json(run_root/'transitions.json',dict(transitions))
            last_report = time.monotonic()
            return result
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
        executor = ProcessPoolExecutor(max_workers=min(args.workers,len(jobs)),
                                       mp_context=multiprocessing.get_context('spawn'),initializer=worker_init)
        try:
            progress('running')
            while pending or not exhausted:
                while not exhausted and len(pending)<2*args.workers:
                    try:job = next(iterator)
                    except StopIteration:exhausted=True;break
                    key = job['dataset']+'/'+job['key']
                    signature = input_signature(job)
                    row = db.execute('SELECT signature,record FROM completed WHERE job_key=?',(key,)).fetchone()
                    if row and row[0] == signature:
                        record = json.loads(row[1])
                        cached_token = record['metadata'].get('source_audit_token')
                        if (all(file_token(token[0]) == token for token in record['output_tokens'])
                                and all(file_token(token[0]) == token for token in record['metadata']['scene_tokens'])
                                and (not cached_token or file_token(cached_token[0]) == cached_token)):
                            account(job,record['metadata'],fresh=False);finished+=1;resumed+=1;continue
                    future = executor.submit(process_group,job,str(run_root),settings)
                    pending[future] = (job,key)
                if not pending:continue
                done,_ = wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    job,key = pending.pop(future)
                    try:
                        result = future.result();tokens = publish(job,result,run_root)
                        record = {'metadata':result['metadata'],'output_tokens':tokens}
                        db.execute('INSERT OR REPLACE INTO completed VALUES (?,?,?)',(key,result['signature'],json.dumps(record)))
                        db.commit();account(job,result['metadata']);finished+=1
                        print(f"{finished}/{len(jobs)} {key}: {result['metadata']['counts']}",flush=True)
                    except Exception as exc:
                        failed+=1
                        with (run_root/'errors.jsonl').open('a') as stream:
                            stream.write(json.dumps({'group':key,'error':f'{type(exc).__name__}: {exc}'})+'\n')
                        print(f'ERROR {key}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
                if time.monotonic()-last_report>10:progress('running')
            if full_selection:
                for key,spec in specifications.items():
                    actual = dataset_counts[key]
                    if actual['input_rows'] != spec['expected_input_rows'] or actual['input_files'] != spec['expected_input_files']:
                        failed+=1;print(f'ERROR count mismatch for {key}: {dict(actual)} expected {spec}',flush=True)
            result = progress('complete' if not failed else 'incomplete')
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            return 0 if not failed else 1
        except BaseException:
            progress('interrupted')
            for process in list(executor._processes.values()):process.terminate()
            raise
        finally:
            executor.shutdown(wait=True,cancel_futures=True);db.close()


def main(argv=None):
    args = parse_args(argv)
    rules = ContactRules.load(REPO_ROOT/'assets/hand/shadow/init_tmpl')
    # Publication uses the same immutable taxonomy ordering as each worker.
    global _WORKER_RULES
    _WORKER_RULES = rules
    jobs,specifications = discover_jobs(args,rules)
    print(json.dumps({'jobs':len(jobs),'sources':specifications,'workers':args.workers},ensure_ascii=False),flush=True)
    if args.dry_run:return 0
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    return run(args,jobs,specifications,rules)


if __name__ == '__main__':
    raise SystemExit(main())
