#!/usr/bin/env python3
"""
startup_drl_truth.py — 运行时 DRL 权威真相

目的: 回答一个问题——"DRL 到底在什么 authority level?"
不看配置, 不看文档, 不看 memory, 只看**运行起来之前**的真相。

用法:
    python scripts/startup_drl_truth.py

会打印:
    1. 所有 config 层面说 DRL 是什么
    2. 实例化之后 self.drl_agent (或等价对象) 的真实 authority
    3. 两者 diff → 这里就是 bug 住的地方

这个脚本不改系统任何东西。纯 diagnostic。
"""
import sys
import os
import re
import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 让 import 在项目根下工作
ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == 'scripts' else Path.cwd()
sys.path.insert(0, str(ROOT))


def section(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------
# Step 1: Collect all config-level DRL declarations
# ---------------------------------------------------------------

def collect_drl_config_declarations() -> List[Dict[str, Any]]:
    """扫描 configs/, .env, 源码里的 DRL_* 声明"""
    declarations = []

    # Scan configs directory
    for cfg_dir in ['configs', 'config']:
        cdir = ROOT / cfg_dir
        if not cdir.exists():
            continue
        for cf in cdir.rglob('*'):
            if not cf.is_file():
                continue
            if cf.suffix not in ('.py', '.json', '.yaml', '.yml', '.env'):
                continue
            try:
                text = cf.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                # Match DRL_MODE, DRL_AUTHORITY, DRL_ENABLED, etc.
                m = re.search(r'\b(DRL_\w+|drl_mode|drl_authority|drl_enabled)\s*[:=]\s*([^#\n]+)', line)
                if m:
                    declarations.append({
                        'file': str(cf.relative_to(ROOT)),
                        'line': line_no,
                        'key': m.group(1),
                        'value': m.group(2).strip().rstrip(',').strip("'\""),
                    })

    # Scan .env
    env_file = ROOT / '.env'
    if env_file.exists():
        try:
            for line_no, line in enumerate(env_file.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
                m = re.match(r'^(DRL_\w+)\s*=\s*(.+)$', line.strip())
                if m:
                    declarations.append({
                        'file': '.env',
                        'line': line_no,
                        'key': m.group(1),
                        'value': m.group(2).strip().strip("'\""),
                    })
        except Exception:
            pass

    # Scan source: assignments to DRL-related variables
    for src_dir in ['defense', 'drl', 'core', 'integration', 'agents', 'v36']:
        sdir = ROOT / src_dir
        if not sdir.exists():
            continue
        for sf in sdir.rglob('*.py'):
            try:
                text = sf.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                m = re.match(r'^(DRL_\w+)\s*=\s*["\']?(\w+)["\']?', line.strip())
                if m:
                    declarations.append({
                        'file': str(sf.relative_to(ROOT)),
                        'line': line_no,
                        'key': m.group(1),
                        'value': m.group(2),
                    })
                m = re.search(r'self\.(drl_authority|drl_mode|drl_enabled|drl_level)\s*=\s*([^#\n]+)', line)
                if m:
                    declarations.append({
                        'file': str(sf.relative_to(ROOT)),
                        'line': line_no,
                        'key': f'self.{m.group(1)}',
                        'value': m.group(2).strip().rstrip(',').strip("'\""),
                    })

    return declarations


# ---------------------------------------------------------------
# Step 2: Try to load the actual DRL agent and inspect it
# ---------------------------------------------------------------

def find_drl_agent_class() -> List[Tuple[str, str]]:
    candidates = []
    patterns = ['drl_agent', 'crypto_drl', 'sb3_drl', 'authority_gate', 'promotion_gate']
    for src_dir in ['agents', 'drl', 'defense']:
        sdir = ROOT / src_dir
        if not sdir.exists():
            continue
        for sf in sdir.rglob('*.py'):
            name = sf.stem
            if not any(p in name for p in patterns):
                continue
            try:
                text = sf.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            for line in text.splitlines():
                m = re.match(r'^class\s+(\w+)', line)
                if m:
                    cls_name = m.group(1)
                    if any(kw in cls_name.lower() for kw in ['drl', 'authority', 'gate', 'lifecycle']):
                        rel = sf.relative_to(ROOT)
                        module_path = str(rel.with_suffix('')).replace('/', '.').replace('\\', '.')
                        candidates.append((module_path, cls_name))
    return candidates


def try_import_and_inspect(module_path: str, class_name: str) -> Dict[str, Any]:
    result = {
        'module': module_path,
        'class': class_name,
        'import_ok': False,
        'instantiate_ok': False,
        'authority_attrs': {},
        'error': None,
    }
    try:
        mod = importlib.import_module(module_path)
        result['import_ok'] = True
        cls = getattr(mod, class_name, None)
        if cls is None:
            result['error'] = f'class {class_name} not in module'
            return result

        try:
            inst = cls()
            result['instantiate_ok'] = True
            for attr in ['authority', 'authority_level', 'mode', 'drl_mode',
                         'current_level', '_state', 'level', 'enabled', 'is_shadow',
                         'is_active', 'is_live', '_authority_level']:
                if hasattr(inst, attr):
                    val = getattr(inst, attr)
                    result['authority_attrs'][attr] = repr(val)[:200]
            for meth in ['get_authority_level', 'get_authority', 'is_shadow_mode']:
                if hasattr(inst, meth) and callable(getattr(inst, meth)):
                    try:
                        val = getattr(inst, meth)()
                        result['authority_attrs'][f'{meth}()'] = repr(val)[:200]
                    except Exception as e:
                        result['authority_attrs'][f'{meth}()'] = f'<ERR: {e}>'
        except Exception as e:
            result['error'] = f'instantiate failed: {e}'
            for attr in ['DEFAULT_AUTHORITY', 'DEFAULT_MODE', 'AUTHORITY', 'VALID_LEVELS']:
                if hasattr(cls, attr):
                    result['authority_attrs'][f'class.{attr}'] = repr(getattr(cls, attr))[:200]
    except Exception as e:
        result['error'] = f'import failed: {type(e).__name__}: {e}'

    return result


# ---------------------------------------------------------------
# Step 3: Check main.py instantiation — where does self.drl_agent get set?
# ---------------------------------------------------------------

def scan_main_for_drl_instantiation() -> List[Dict[str, Any]]:
    findings = []
    for entry in ['main.py', 'app.py', 'run.py', 'entrypoint.py']:
        mp = ROOT / entry
        if not mp.exists():
            continue
        try:
            text = mp.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            m = re.search(r'self\.(drl\w*|_drl\w*)\s*=\s*(.+)', line)
            if m:
                findings.append({
                    'file': entry,
                    'line': line_no,
                    'attr': m.group(1),
                    'assignment': m.group(2).strip()[:120],
                })
            m = re.search(r'(authority|mode|drl_authority)\s*=\s*["\'](\w+)["\']', line)
            if m and 'drl' in line.lower():
                findings.append({
                    'file': entry,
                    'line': line_no,
                    'attr': m.group(1),
                    'assignment': f'"{m.group(2)}"',
                })
    return findings


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("  DRL AUTHORITY TRUTH  |  what is DRL *really* running as?")
    print("=" * 72)

    # --- 1. Config declarations ---
    section("1. All DRL-related declarations in configs + source")
    decls = collect_drl_config_declarations()
    if not decls:
        print("  (no DRL-related declarations found)")
    else:
        print(f"  Found {len(decls)} declarations:")
        by_key: Dict[str, List[Dict]] = {}
        for d in decls:
            by_key.setdefault(d['key'], []).append(d)

        for key in sorted(by_key.keys()):
            entries = by_key[key]
            vals = {e['value'] for e in entries}
            status = "[!] CONFLICT" if len(vals) > 1 else "    OK      "
            print(f"\n  {status} {key}:")
            for e in entries:
                print(f"       {e['value']!r}   at  {e['file']}:{e['line']}")

        print()
        print("  -> Any CONFLICT above = bug root cause")

    # --- 2. main.py instantiation ---
    section("2. main.py self.drl_* assignments")
    findings = scan_main_for_drl_instantiation()
    if not findings:
        print("  (no self.drl_* assignments found — DRL may be instantiated")
        print("   via factory / lazy / in a different module)")
    else:
        for f in findings:
            print(f"  {f['file']}:{f['line']:<5} self.{f['attr']:<25} = {f['assignment']}")

    # --- 3. Actual class inspection ---
    section("3. Import DRL-related classes and inspect their default state")
    candidates = find_drl_agent_class()
    print(f"  Found {len(candidates)} candidate classes:")
    for mod, cls in candidates:
        print(f"    {mod}.{cls}")

    print()
    for mod, cls in candidates:
        result = try_import_and_inspect(mod, cls)
        print(f"\n  -- {mod}.{cls} --")
        print(f"    import: {'OK' if result['import_ok'] else 'FAIL'}")
        if result['import_ok']:
            print(f"    instantiate (no args): {'OK' if result['instantiate_ok'] else 'FAIL'}")
            if result['authority_attrs']:
                print(f"    attributes found:")
                for k, v in result['authority_attrs'].items():
                    print(f"      {k} = {v}")
            elif result['instantiate_ok']:
                print(f"    (no authority-like attributes)")
        if result['error']:
            print(f"    error: {result['error']}")

    # --- 4. Verdict template ---
    section("4. Fill in manually after reading above")
    print("""
  Config layer says DRL should be:      ___________________
  After instantiation self.drl_agent:   ___________________
  Diff (where the bug is):              ___________________
""")


if __name__ == '__main__':
    main()
