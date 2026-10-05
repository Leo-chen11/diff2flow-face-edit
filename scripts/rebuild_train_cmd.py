"""Rebuild the exact train_sdflow command of a finished run from its
config.json, optionally changing a few flags -- so a follow-up experiment
differs from its parent run ONLY in the flags you name.

train_sdflow.py writes vars(args) to ./output/<model_name>/<run_name>/config.json.
This script reads the trainer's own argparse definitions (without importing
torch), turns that file back into CLI flags, and prints a runnable command.
By default only flags that differ from the trainer's defaults are printed.

Usage:
    python -m scripts.rebuild_train_cmd \
        --config ./output/SDFlow/multi_v1_cont20k_ctrl/config.json \
        --set run_name=multi_v1_cont20k_next \
        --set clip_prompt_weight=0
"""
import argparse
import ast
import json
import os
import shlex
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'training', 'train_sdflow.py')


def load_train_parser(path=TRAIN_SCRIPT):
    """Execute only the argparse statements of train_sdflow's __main__ block
    (everything before `args = parser.parse_args()`)."""
    tree = ast.parse(open(path).read())
    main = next(n for n in tree.body
                if isinstance(n, ast.If) and '__main__' in ast.dump(n.test))
    stmts = []
    for s in main.body:
        if isinstance(s, ast.Assign) and 'parse_args' in ast.dump(s.value):
            break
        stmts.append(s)
    ns = {'argparse': argparse}
    exec(compile(ast.Module(body=stmts, type_ignores=[]), path, 'exec'), ns)
    return ns['parser']


def _as_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('1', 'true', 'yes', 'on'):
        return True
    if s in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'not a boolean: {v!r}')


def _is_bool_action(action):
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                               argparse.BooleanOptionalAction))


def _parse_override(action, raw):
    if _is_bool_action(action):
        return _as_bool(raw)
    if raw.lower() in ('none', 'null'):
        return None
    if action.nargs not in (None, '?'):
        items = raw.split()
        return [action.type(x) if action.type else x for x in items]
    return action.type(raw) if action.type else raw


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def to_flags(action, value):
    """CLI tokens that make argparse produce `value` for this action, or []
    when the action should be left out."""
    opt = next(o for o in action.option_strings if o.startswith('--'))
    if isinstance(action, argparse.BooleanOptionalAction):
        name = opt[2:]
        return [opt] if _as_bool(value) else [f'--no-{name}']
    if isinstance(action, argparse._StoreTrueAction):
        return [opt] if _as_bool(value) else []
    if isinstance(action, argparse._StoreFalseAction):
        return [] if _as_bool(value) else [opt]
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if not value and action.nargs == '+':
            return []
        return [opt] + [str(x) for x in value]
    return [opt, str(value)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='config.json of the parent run.')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='Change one flag (repeatable). Lists: space-separated in quotes. '
                        'Booleans: true/false. VALUE none drops the flag.')
    p.add_argument('--unset', action='append', default=[], metavar='KEY',
                   help='Reset a flag to the trainer default (repeatable).')
    p.add_argument('--all', action='store_true',
                   help='Print every flag, not only those that differ from the defaults.')
    p.add_argument('--prefix', default='python training/train_sdflow.py')
    args = p.parse_args()

    parser = load_train_parser()
    actions = {a.dest: a for a in parser._actions if a.option_strings and a.dest != 'help'}
    cfg = json.load(open(args.config))

    unknown = sorted(k for k in cfg if k not in actions)
    if unknown:
        print(f'# ignored (not trainer flags anymore): {" ".join(unknown)}', file=sys.stderr)

    values = {k: cfg[k] for k in cfg if k in actions}
    changes = []
    for item in args.set:
        key, sep, raw = item.partition('=')
        key = key.strip().lstrip('-').replace('-', '_')
        if not sep or key not in actions:
            raise SystemExit(f'--set {item!r}: unknown trainer flag {key!r}')
        new = _parse_override(actions[key], raw)
        changes.append((key, values.get(key, actions[key].default), new))
        values[key] = new
    for key in args.unset:
        key = key.strip().lstrip('-').replace('-', '_')
        if key not in actions:
            raise SystemExit(f'--unset {key!r}: unknown trainer flag')
        changes.append((key, values.get(key), actions[key].default))
        values.pop(key, None)

    tokens = []
    for dest, action in actions.items():
        if dest not in values:
            continue
        value = values[dest]
        if not args.all and _norm(value) == _norm(action.default):
            continue
        tokens.append(to_flags(action, value))

    # Round-trip check: the printed flags must parse back to exactly `values`.
    parsed = vars(parser.parse_args([t for group in tokens for t in group]))
    bad = [k for k, v in values.items() if _norm(parsed.get(k)) != _norm(v)]
    if bad:
        raise SystemExit(f'round-trip mismatch for: {bad}')

    for key, old, new in changes:
        print(f'# {key}: {old!r} -> {new!r}', file=sys.stderr)
    lines = [args.prefix] + ['  ' + ' '.join(shlex.quote(t) for t in g) for g in tokens if g]
    print(' \\\n'.join(lines))


if __name__ == '__main__':
    main()
