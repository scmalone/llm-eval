"""Immutable paired campaigns over the existing single-run restore/resume boundary.

Each attempt gets a fresh session and a cold scratch service, including after interruption.
The scheduler owns no services: execute() owns locks, inhibition, cleanup and restoration.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from .plan import fingerprint, verify_plan
from .state import atomic_json, machine_lock
from .metrics.ingest import task_identity_sha

SEEDS = (17, 29, 43)


def _leaves(value, path=''):
    if isinstance(value, dict) and value:
        return {k: v for key, child in value.items() for k, v in
                _leaves(child, path + ('.' if path else '') + key).items()}
    return {path: value}


def _contract(plan):
    if len(plan['models']) != 1:
        raise ValueError('paired arms need exactly one model')
    model = plan['models'][0]
    work = model.get('workload')
    if not work or any(f['kind'] != 'workload' for f in model['fixtures']):
        raise ValueError('paired arms need workload-only plans')
    context = work['context']
    if model['contexts'] != [context] or model['anchor'] != context:
        raise ValueError('paired arms need one fixed context; step-down is a separate cohort')
    settings = copy.deepcopy(model['settings'])
    runtime = plan.get('runtime', 'llama.cpp')
    # Strata's settings.pinned_files includes executable/code/model/template hashes. A llama.cpp profile's
    # settings are its preset (server flags); its pinned request sits beside them at the model level.
    return {'backend': plan['backend'], 'runtime': runtime, 'model': model['id'],
            'request': copy.deepcopy(model.get('request')) if runtime != 'strata' else None,
            'settings': settings, 'limits': plan['limits'],
            'context': context, 'mode': work.get('mode'),
            'client': {'harness': work.get('harness', 'opencode'),
                       'sha256': plan['files'][work['binary' if work.get('harness') == 'pi' else 'opencode_binary']],
                       'version': work['binary_version' if work.get('harness') == 'pi' else 'opencode_version'],
                       'config': work['config']},
            'tasks': {t['id']: task_identity_sha(t, plan['files']) for t in model['fixtures']}}


# What a model comparison may never change between arms: the tasks, the context, the harness and its version.
# The rest of a contract (runtime, model, settings, request, the client's model entry, limits) is the model.
MODEL_INVARIANT = ('tasks.', 'context', 'mode.context', 'client.harness', 'client.sha256', 'client.version')


def freeze(arms, directory, allowed_changes=(), kind='settings', seeds=SEEDS):
    """Freeze two already prepared/pinned plans. Refuse undeclared changes before any run.

    ``seeds``: the paired seeds (default all three; the check suite's gate runs one).
    ``kind='settings'``: one model, one runtime, arms that differ only in declared ``settings.*`` leaves (an
    engine knob or a llama.cpp preset flag). ``kind='models'``: two models (any runtime) on the same tasks,
    context and harness; every difference is recorded, none of the invariants may move."""
    if set(arms) != {'A', 'B'}:
        raise ValueError('exactly arms A and B are required')
    if kind not in ('settings', 'models'):
        raise ValueError("kind must be 'settings' or 'models'")
    for plan in arms.values():
        verify_plan(plan)
    contracts = {arm: _contract(plan) for arm, plan in arms.items()}
    leaves = {arm: _leaves(contract) for arm, contract in contracts.items()}
    changes = {key: {'A': leaves['A'].get(key), 'B': leaves['B'].get(key)}
               for key in sorted(leaves['A'].keys() | leaves['B'].keys())
               if leaves['A'].get(key) != leaves['B'].get(key)}
    if kind == 'models':
        moved = [key for key in changes if key.startswith(MODEL_INVARIANT)]
        if moved:
            raise ValueError(f'a model comparison must keep tasks, context and harness: {moved[:6]}')
        if contracts['A']['model'] == contracts['B']['model'] and contracts['A']['runtime'] == contracts['B']['runtime']:
            raise ValueError('a model comparison needs two different models; a knob is kind settings')
        allowed_changes = sorted(changes)
    elif contracts['A']['runtime'] != contracts['B']['runtime']:
        raise ValueError('a settings campaign keeps one runtime; two runtimes are kind models')
    # Output budgets, fixtures, client and context cannot drift within a settings cohort.
    if kind == 'settings' and any(not key.startswith('settings.') for key in allowed_changes):
        raise ValueError('only explicit settings.* changes can be declared in this cohort')
    undeclared = changes.keys() - set(allowed_changes)
    if undeclared:
        raise ValueError(f'undeclared arm changes: {sorted(undeclared)}')
    tasks = sorted(contracts['A']['tasks'])
    attempts = []
    for task in tasks:
        for pair, seed in enumerate(seeds, 1):
            for arm in ('A', 'B') if pair % 2 else ('B', 'A'):
                attempts.append({'id': f'{task}-p{pair}-{arm.lower()}', 'pair': f'{task}-p{pair}',
                                 'task': task, 'seed': seed, 'arm': arm})
    manifest = {'version': 1, 'kind': kind, 'arms': copy.deepcopy(arms), 'allowed_changes': sorted(allowed_changes),
                'changes': changes, 'seeds': list(seeds), 'attempts': attempts,
                'reset': 'fresh_session_and_cold_service_per_attempt', 'retries': 0}
    manifest['campaign_hash'] = fingerprint(manifest)
    path = Path(directory)/'campaign.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('campaign exists with a different contract; use a new directory')
    else:
        atomic_json(path, manifest)
    return manifest


def verify(manifest):
    if fingerprint({k: v for k, v in manifest.items() if k != 'campaign_hash'}) != manifest['campaign_hash']:
        raise ValueError('campaign integrity mismatch')
    for plan in manifest['arms'].values():
        verify_plan(plan)


def attempt_plan(manifest, attempt):
    plan = copy.deepcopy(manifest['arms'][attempt['arm']])
    model = plan['models'][0]
    model['fixtures'] = [{**f, 'runs': 1} for f in model['fixtures'] if f['id'] == attempt['task']]
    if len(model['fixtures']) != 1:
        raise ValueError('attempt task missing from arm')
    if plan.get('runtime', 'llama.cpp') == 'strata':
        model['settings']['request']['seed'] = attempt['seed']
    else:
        # llama.cpp: the request is the model's, beside the preset; settings stay server flags only
        model['request'] = {**(model.get('request') or {}), 'seed': attempt['seed']}
    plan['paired_attempt'] = {**attempt, 'campaign_hash': manifest['campaign_hash']}
    plan['plan_hash'] = fingerprint({k: v for k, v in plan.items() if k != 'plan_hash'})
    return plan


def _outcome(state, attempt, elapsed):
    trials = [v for v in state['sections'].values() if v.get('task') == attempt['task']]
    trial = next(iter(trials), {})
    qualification = next((v for k, v in state['sections'].items() if k.endswith('__context_qualification')), {})
    loads = qualification.get('attempts', [])
    return {**attempt, 'status': trial.get('status', 'infra_fail'), 'elapsed_seconds': elapsed,
            'category': trial.get('cohort_category') or trial.get('category'),
            'load_seconds': sum(a['load_seconds'] for a in loads if a.get('load_seconds') is not None) if loads else None,
            'context': trial.get('context'), 'stop': trial.get('stop'),
            'metrics': trial.get('metrics', {}), 'run_status': state['status']}


def run(directory, approve, resume=False, executor=None, adapter_factory=None):
    from .runner import execute
    from .runtime import adapter_class
    from . import integration
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'campaign.json').read_text())
    if approve != manifest['campaign_hash'][:12]:
        raise PermissionError('approval token does not match frozen campaign')
    # Serialize even between attempts so concurrent schedulers cannot interleave the order.
    with machine_lock(directory/'campaign.lock'):
        if fingerprint({k: v for k, v in manifest.items() if k != 'campaign_hash'}) != manifest['campaign_hash']:
            raise ValueError('campaign integrity mismatch')
        state_path = directory/'paired-state.json'
        if state_path.exists() and not resume:
            raise ValueError('campaign already started; use resume')
        saved = json.loads(state_path.read_text()) if state_path.exists() else {'completed': {}, 'elapsed': {}}
        if saved.get('campaign_hash', manifest['campaign_hash']) != manifest['campaign_hash']:
            raise ValueError('saved campaign hash mismatch')
        saved['campaign_hash'] = manifest['campaign_hash']
        if resume:
            saved.pop('halted', None)
        # An interrupted attempt must recover through execute before input verification: it
        # restores captured production first, then refuses a changed input under its own lock.
        pending = saved.get('pending')
        if pending is None:
            verify(manifest)
        if manifest['arms']['A']['backend'] == 'live':
            why = integration.blocker()
            if why and pending is None:
                raise PermissionError(why)
        for attempt in manifest['attempts']:
            identifier = attempt['id']
            if identifier in saved['completed']:
                continue
            plan = attempt_plan(manifest, attempt)
            run_id = 'paired-' + manifest['campaign_hash'][:12] + '-' + identifier
            run_dir = directory/'runs'/run_id
            saved['pending'] = identifier
            atomic_json(state_path, saved)
            adapter = None
            if adapter_factory:
                adapter = adapter_factory(plan, run_dir, run_id)
            elif plan['backend'] == 'live':
                adapter = adapter_class(plan)(plan, run_dir/'server-logs', run_id=run_id)
            before = time.monotonic()
            try:
                state = (executor or execute)(plan, directory/'runs', run_id,
                    resume=(run_dir/'state.json').exists(), adapter=adapter)
            finally:
                saved['elapsed'][identifier] = saved['elapsed'].get(identifier, 0) + time.monotonic() - before
                atomic_json(state_path, saved)
            if state.get('restoration_required'):
                raise RuntimeError('STOP: production not verified restored; recover pending attempt before continuing')
            if state['status'] != 'complete':
                return {**report(manifest, saved), 'status': 'interrupted', 'pending': identifier}
            outcome = _outcome(state, attempt, saved['elapsed'][identifier])
            saved['completed'][identifier] = outcome
            saved.pop('pending', None)
            atomic_json(state_path, saved)
            verify(manifest)
            if outcome['status'] in ('infra_fail', 'capacity_fail'):
                # Infrastructure failure remains a completed attempt with visible cost. Do
                # not blindly carry on or retry it as a fresh first attempt.
                saved['halted'] = identifier
                atomic_json(state_path, saved)
                break
        result = report(manifest, saved)
        atomic_json(directory/'paired-report.json', result)
        return result


def report(manifest, saved):
    completed = saved['completed']
    paired, common_time, lost, gained = [], {'A': 0.0, 'B': 0.0}, [], []
    for pair in dict.fromkeys(a['pair'] for a in manifest['attempts']):
        rows = {a['arm']: completed[a['id']] for a in manifest['attempts']
                if a['pair'] == pair and a['id'] in completed}
        both = len(rows) == 2
        common = both and all(r['status'] == 'pass' for r in rows.values())
        if common:
            for arm, row in rows.items():
                common_time[arm] += row['elapsed_seconds']
        paired.append({'pair': pair, 'complete': both, 'common_success': common, 'arms': rows})
        if both and (rows['A']['status'] == 'pass') != (rows['B']['status'] == 'pass'):
            (lost if rows['A']['status'] == 'pass' else gained).append(pair)
    arms = {}
    for arm in ('A', 'B'):
        rows = [r for r in completed.values() if r['arm'] == arm]
        arms[arm] = {'recorded': len(rows), 'passed': sum(r['status'] == 'pass' for r in rows),
                     'timeouts': sum(r['status'] == 'timeout' for r in rows),
                     'infrastructure_failures': sum(r['status'] in ('infra_fail', 'capacity_fail') for r in rows),
                     'all_attempt_seconds': sum(saved.get('elapsed', {}).get(a['id'], 0)
                                                for a in manifest['attempts'] if a['arm'] == arm),
                     'common_success_seconds': common_time[arm],
                     'categories': {c: {'recorded': sum(r.get('category') == c for r in rows),
                         'passed': sum(r.get('category') == c and r['status'] == 'pass' for r in rows)}
                         for c in sorted({r.get('category') for r in rows if r.get('category')})}}
    return {'campaign_hash': manifest['campaign_hash'], 'complete': len(completed) == len(manifest['attempts']),
            'planned': len(manifest['attempts']), 'recorded': len(completed), 'arms': arms, 'pairs': paired,
            'lost_by_b': lost, 'gained_by_b': gained, 'seeds': manifest.get('seeds', list(SEEDS)),
            'halted': saved.get('halted'), 'retry_policy': 'none; first attempt only',
            'uncertainty': f"{len(manifest.get('seeds', SEEDS))} paired seed(s) are a screen, not statistical "
                           'equivalence'}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python3 -m llm_eval.paired')
    commands = parser.add_subparsers(dest='command', required=True)
    freeze_cmd = commands.add_parser('freeze')
    freeze_cmd.add_argument('--arm-a', required=True)
    freeze_cmd.add_argument('--arm-b', required=True)
    freeze_cmd.add_argument('--allow', action='append', default=[])
    freeze_cmd.add_argument('--kind', default='settings', choices=('settings', 'models'))
    freeze_cmd.add_argument('--seeds', type=int, default=len(SEEDS), choices=range(1, len(SEEDS) + 1),
                            help='paired seeds (1 is the check suite gate)')
    for name in ('run', 'resume'):
        commands.add_parser(name).add_argument('--approve', required=True)
    commands.add_parser('status')
    for command in commands.choices.values():
        command.add_argument('--dir', required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        manifest = freeze({arm: json.loads(Path(path).read_text()) for arm, path in
                           (('A', args.arm_a), ('B', args.arm_b))}, args.dir, args.allow, args.kind,
                          SEEDS[:args.seeds])
        result = {'planned': len(manifest['attempts']), 'approval_token': manifest['campaign_hash'][:12],
                  'changes': manifest['changes']}
    elif args.command == 'status':
        manifest = json.loads((Path(args.dir)/'campaign.json').read_text())
        state = Path(args.dir)/'paired-state.json'
        result = report(manifest, json.loads(state.read_text()) if state.exists() else {'completed': {}})
    else:
        result = run(args.dir, args.approve, args.command == 'resume')
    print(json.dumps(result, indent=2))
    return 0 if result.get('complete', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
