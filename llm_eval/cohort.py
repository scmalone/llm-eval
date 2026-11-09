"""Freeze the separate 18-task Strata roster without changing the legacy 44 tasks.

Document prompts are sized once, offline with the installed tokenizer and template,
then pinned as literal prompts. Both arms use the same bytes, even across engines.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import contextlib
import subprocess
import select

from .haystack import filler_block
from .plan import digest
from .state import atomic_json
from .workload.prepare import TASK_DIR, load_tasks

ROSTER = Path(__file__).resolve().parents[1]/'fixtures/strata/cohort.json'
DEPTH_ROSTER = Path(__file__).resolve().parents[1]/'fixtures/strata/depth.json'


@contextlib.contextmanager
def runtime_counter(python, config, server):
    """Use the installed Strata Python's dependencies without changing either environment."""
    worker = subprocess.Popen([str(python), '-u', '-m', 'llm_eval.cohort', str(config), str(server)],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    def receive():
        if not select.select([worker.stdout], [], [], 120)[0]:
            raise ValueError('offline Strata token counter timed out')
        line = worker.stdout.readline()
        if not line:
            raise ValueError('offline Strata token counter failed: ' + worker.stderr.read()[-1000:])
        body = json.loads(line)
        if 'error' in body:
            raise ValueError('offline Strata token counter failed: ' + body['error'])
        return body
    try:
        receive()
        def count(text):
            worker.stdin.write(json.dumps(text) + '\n')
            worker.stdin.flush()
            return receive()['tokens']
        yield count
    finally:
        worker.terminate()
        try:
            worker.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate()


LLAMA_TOKENIZE = Path("~/Dev/llama.cpp/build/bin/llama-tokenize").expanduser()  # the pinned build's (build 12017)


@contextlib.contextmanager
def llama_counter(model_path, binary=LLAMA_TOKENIZE):
    """Count a prompt's tokens with a llama.cpp profile's own vocabulary (``llama-tokenize`` loads the GGUF's
    vocabulary only, on the CPU), so a 32K document is 32K for every model, not just for Strata's tokenizer."""
    model = Path(model_path).expanduser()
    if not Path(binary).is_file() or not model.is_file():
        raise ValueError(f"cannot count tokens: {binary if not Path(binary).is_file() else model} is missing")

    def count(text):
        done = subprocess.run([str(binary), "-m", str(model), "--stdin", "--show-count", "--log-disable", "--ids"],
                              input=text, capture_output=True, text=True, timeout=300)
        found = [line for line in done.stdout.splitlines() if line.startswith("Total number of tokens:")]
        if done.returncode or not found:
            raise ValueError("llama-tokenize failed: " + (done.stderr or done.stdout)[-500:])
        return int(found[-1].split(":")[1])
    yield count


def documents(directory, count, roster=ROSTER):
    """The six document tasks, sized by ``count`` (this profile's tokenizer) and frozen into ``directory``."""
    spec = json.loads(Path(roster).read_text())
    tasks = [document(doc, count) for doc in spec['document']]
    for task in tasks:
        # llama-tokenize counts the raw user text; Strata's counter (token_counter) counts it rendered in the chat
        # template, a few dozen tokens more (about 0.1% at 32K). The label says which, so the two are never read
        # as the same measurement (red-team question, 2026-11-01).
        task['prompt_measurement']['method'] = 'profile_tokenizer_raw_user_text'
        task['prompt_measurement']['note'] = 'raw user text, no chat template; the template adds a few dozen tokens'
    return _freeze(directory, tasks, roster, spec)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def token_counter(config, server):
    """Use Strata's CPU-only tokenizer/template modules, never its engine or service."""
    config_path = Path(config).resolve()
    cfg = json.loads(config_path.read_text())
    cwd = (config_path.parent/Path(cfg.get('cwd', '.')).expanduser()).resolve()
    tokenizer = (cwd/Path(cfg['tokenizer']).expanduser()).resolve()
    source = Path(server).resolve().parents[1]
    st = _module('_llm_eval_strata_tokenizer', source/'tools/strata_tokenizer.py')
    frontend = _module('_llm_eval_strata_frontend', source/'serve/frontend.py')
    vocab = json.loads((tokenizer/'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for text, index in vocab.items():
        tokens[index] = text
    tok = st.Tokenizer(tokens, (tokenizer/'merges.txt').read_text().split('\n'),
                       json.loads((tokenizer/'token_type.json').read_text()))
    template = frontend.ChatTemplate(tokenizer/'chat_template.jinja')
    return lambda text: len(tok.encode(template.render([{'role': 'user', 'content': text}],
                               reasoning_effort='xhigh'), parse_special=True))


def document(spec, count):
    rng = random.Random(spec['seed'])
    code, arrived, sent = rng.randint(1000, 9000), rng.randint(50, 90), rng.randint(10, 40)
    facts = [f'[R-01] The current approved Kestrel dispatch code is K-{code}.',
             f'[R-02] Kestrel received {arrived} crates. There were no crates before this receipt.',
             f'[R-03] Kestrel dispatched {sent} crates after the receipt. No other crates moved.',
             '[R-04] A draft suggests dispatch code K-9999. Drafts are not approved.',
             '[R-05] The Alder depot manager is Rowan. Kestrel has no manager record in this corpus.']
    if spec['case'] == 'retrieve':
        question = 'What is the current approved Kestrel dispatch code? Cite its record ID.'
        answer = f'K-{code} [R-01]'
        checks = [{'type': 'answer', 'all': [f'K-{code}\\b', r'\[R-01\]'], 'none': ['K-9999'], 'none_scope': 'claim'}]
    elif spec['case'] == 'reason':
        question = 'How many crates remain at Kestrel? Explain the arithmetic and cite both relevant record IDs.'
        answer = f'{arrived} - {sent} = {arrived-sent} crates remain [R-02] [R-03].'
        checks = [{'type': 'answer', 'all': [rf'\b{arrived-sent}\b', r'\[R-02\]', r'\[R-03\]']}]
    else:
        question = 'Who manages Kestrel? If the corpus does not supply a name, say NOT IN RECORD and cite the evidence.'
        answer = 'NOT IN RECORD [R-05]'
        checks = [{'type': 'answer', 'all': ['NOT IN RECORD', r'\[R-05\]'], 'none': ['Rowan'], 'none_scope': 'claim'}]
    def render(blocks):
        slots = {i * (blocks-1)//4: fact for i, fact in enumerate(facts)}
        corpus = '\n'.join((slots.get(i, '') + '\n' + filler_block(spec['seed'], i)) for i in range(blocks))
        return ('Use only the following corpus. Cite bracketed record IDs; never guess missing facts.\n'
                '<corpus>\n' + corpus + '\n</corpus>\n' + question)
    target, blocks, best = spec['tokens'], max(5, spec['tokens']//60), None
    for _ in range(16):
        prompt = render(blocks)
        measured = count(prompt)
        if measured <= target and (best is None or measured > best[1]):
            best = prompt, measured
        if target * .995 <= measured <= target:
            break
        changed = max(5, round(blocks * target / max(measured, 1) * .998))
        blocks = changed if changed != blocks else max(5, blocks + (-1 if measured > target else 1))
    if best is None or best[1] < target * .995:
        raise ValueError('document prompt could not be sized within 0.5% of its target')
    return {'id': spec['id'], 'kind': 'workload', 'category': 'lookup', 'cohort_category': 'document',
            'prompt': best[0], 'source': {'replica': 'document-room', 'seed': spec['seed']},
            'checks': checks + [{'type': 'diff_scope', 'deny': ['**']}],
            'limits': {'turns': 1, 'wall_seconds': 1800}, 'runs': 1, 'solution': {'answer': answer},
            'prompt_measurement': {'tokens': best[1], 'target': target, 'method': 'templated_user_only',
                                   'note': 'actual OpenCode system/tools overhead is captured per request'}}


def depth_task(spec, count):
    """A session that grows to a fraction of the compaction point before the question is asked.

    This is the only shape that measures whether a large context is usable rather than merely installed:
    the fact is stated at the start, the fill reads real files until the session is deep, and the question
    then asks for that first fact. Sized against the run's own compaction point, so the same task means
    the same depth at 131K and at 262K.
    """
    rng = random.Random(spec['seed'])
    code = rng.randint(1000, 9000)
    files = [f'notes-{n}.txt' for n in range(spec.get('files', 4))]
    return {'id': spec['id'], 'kind': 'workload', 'category': 'lookup', 'cohort_category': 'depth',
            'prelude': [f'For this session: the approved Kestrel dispatch code is K-{code} [R-01]. Just say OK.'],
            'fill': {'fraction': spec.get('fraction', 0.9), 'max_turns': spec.get('max_turns', 14),
                     'prompts': [f'Read {name} in full, then give me one line on what it says.' for name in files]},
            'prompt': 'What approved Kestrel dispatch code did I give you at the start of this session? '
                      'Cite its record ID.',
            'source': {'replica': 'document-room', 'seed': spec['seed']},
            'plant': {name: filler_block(spec['seed'], n) * spec.get('blocks', 6)
                      for n, name in enumerate(files)},
            'checks': [{'type': 'answer', 'all': [f'K-{code}\\b', r'\[R-01\]'], 'none': ['K-9999'], 'none_scope': 'claim'},
                       {'type': 'depth', 'min_fraction': spec.get('min_fraction', 0.8)},
                       {'type': 'diff_scope', 'deny': ['**']}],
            'limits': {'turns': 1, 'wall_seconds': spec.get('wall_seconds', 3600)}, 'runs': 1,
            'solution': {'answer': f'K-{code} [R-01]'},
            'prompt_measurement': {'method': 'inline', 'note': 'the fill grows the session; the question is short'}}


def prepare(directory, count, roster=ROSTER, task_dir=TASK_DIR):
    roster = Path(roster)
    spec = json.loads(roster.read_text())
    known = {t['id']: t for t in load_tasks(task_dir)}
    if 'depth' in spec:  # the extra roster: what the 18-task cohort cannot measure
        tasks = [depth_task(doc, count) for doc in spec['depth']]
        return _freeze(directory, tasks, roster, spec)
    tasks = []
    for category in ('coding', 'tool'):
        for task_id in spec[category]:
            task = copy.deepcopy(known[task_id])
            task.update(cohort_category=category, runs=1)
            tasks.append(task)
    tasks += [document(doc, count) for doc in spec['document']]
    if len(tasks) != 18 or len({t['id'] for t in tasks}) != 18:
        raise ValueError('cohort must have 18 unique tasks')
    return _freeze(directory, tasks, roster, spec)


def _freeze(directory, tasks, roster, spec):
    for task in tasks:
        path = Path(directory)/'cohort-tasks'/f"{task['id']}.json"
        clean = {k: v for k, v in task.items() if k != '_path'}
        if path.exists() and json.loads(path.read_text()) != clean:
            raise ValueError('frozen cohort differs; use a new directory')
        atomic_json(path, clean)
        task['_path'] = str(path.resolve())
    atomic_json(Path(directory)/'cohort.json', {'roster': str(roster), 'sha256': digest(roster),
                'output_tokens': spec['output_tokens'], 'retries': spec['retries'],
                'tasks': {t['id']: digest(t['_path']) for t in tasks}})
    return tasks


if __name__ == '__main__':
    try:
        count = token_counter(*sys.argv[1:])
        print(json.dumps({'ready': True}), flush=True)
        for line in sys.stdin:
            print(json.dumps({'tokens': count(json.loads(line))}), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
        raise SystemExit(1)
