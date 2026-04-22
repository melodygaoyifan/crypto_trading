#!/usr/bin/env python3
"""
startup_agent_wiring_truth.py — Agent 接入真相

目的: 回答一个问题——"agents/ 目录里每个 agent 类, 在 main.py 里到底有没有被用?"
不看 memory, 不看审计结论, 只看**文件系统 + AST**.

用法:
    python scripts/startup_agent_wiring_truth.py

会打印:
    1. agents/ 目录实际有哪些 class (AST 枚举)
    2. 每个 class 在 main.py 里的 4 级接入状态:
       L1: imported          (import 了)
       L2: instantiated      (调了 constructor)
       L3: attribute set     (self.xxx = ...)
       L4: method called     (self.xxx.method(...))
    3. 最后一列是综合 verdict: ACTIVE / PARTIAL / DEAD

不改系统任何东西。纯 diagnostic。
"""
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == 'scripts' else Path.cwd()


def enumerate_agent_classes() -> List[Dict[str, Any]]:
    agents_dir = ROOT / 'agents'
    if not agents_dir.exists():
        print(f"  [ERR] agents/ does not exist at {ROOT}")
        return []

    classes = []
    for fpath in sorted(agents_dir.rglob('*.py')):
        if '__pycache__' in fpath.parts or 'archive' in fpath.parts or 'legacy' in fpath.parts:
            continue
        if fpath.name == '__init__.py':
            continue
        try:
            source = fpath.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(source)
        except Exception as e:
            classes.append({
                'file': str(fpath.relative_to(ROOT)),
                'class': '<PARSE_FAILED>',
                'lines': 0,
                'error': str(e),
            })
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({
                    'file': str(fpath.relative_to(ROOT)),
                    'class': node.name,
                    'line': node.lineno,
                    'methods': len(methods),
                    'file_lines': len(source.splitlines()),
                })

    return classes


def find_main_entry() -> Path:
    for cand in ['main.py', 'app.py', 'run.py', 'entrypoint.py']:
        p = ROOT / cand
        if p.exists():
            return p
    return None


def _normalize_imports(src: str) -> str:
    """Collapse multi-line `from X import (A, B, C)` blocks onto one line."""
    # Match `from X import (\n  A,\n  B,\n)` and flatten
    return re.sub(
        r'(from\s+[\w.]+\s+import\s*)\(\s*([^)]+?)\s*\)',
        lambda m: m.group(1) + ' '.join(x.strip() for x in m.group(2).split(',')),
        src,
        flags=re.DOTALL,
    )


def _smart_snake(s: str) -> str:
    """PascalCase/CamelCase → snake_case, handling initialisms.

    LLMSentimentAgent → llm_sentiment_agent (not l_l_m_sentiment_agent)
    KrakenQuantAgentV6 → kraken_quant_agent_v6
    """
    # Insert underscore between lowercase/digit → uppercase: aB → a_B
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Insert underscore between uppercase-block → uppercase+lowercase: ABCd → AB_Cd
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return s.lower()


def _discover_aliases(main_src: str, cls_name: str, module_path: str,
                      extra_factories: List[str] = None) -> List[str]:
    """Find all names that reference this class in main_src.

    Covers:
      from X import ClassName                       -> [ClassName]
      from X import ClassName as _Alias             -> [ClassName, _Alias]
      from agents import ClassName [as _Alias]      -> [...]
      get_classname() factory wrapping              -> [get_classname]
      from X import ( ClassName, ... )              -> [ClassName] (multi-line)
      extra_factories                                -> [factory_name, alias_of_factory, ...]
    """
    # Normalize multi-line imports first so single-line regex catches them
    main_src = _normalize_imports(main_src)
    names = {cls_name}

    # Aliased direct imports: `from <module> import Foo as _Bar, Baz as _Qux`
    for m in re.finditer(
        rf'from\s+{re.escape(module_path)}\s+import\s+([^\n#]+)', main_src
    ):
        for item in m.group(1).split(','):
            item = item.strip()
            parts = re.split(r'\s+as\s+', item)
            if len(parts) == 2 and parts[0].strip() == cls_name:
                names.add(parts[1].strip())
            elif parts[0].strip() == cls_name:
                pass  # already in names

    # Package re-exports: `from agents import X as _Y` or `from agents.foo import X as _Y`
    for m in re.finditer(r'from\s+agents[\w.]*\s+import\s+([^\n#]+)', main_src):
        for item in m.group(1).split(','):
            item = item.strip()
            parts = re.split(r'\s+as\s+', item)
            if len(parts) == 2 and parts[0].strip() == cls_name:
                names.add(parts[1].strip())

    # Factory functions: get_<snake_case>, create_<snake_case>
    _snake = _smart_snake  # smart initialism handling

    factory_names: List[str] = []
    snake = _snake(cls_name)
    for prefix in ('get_', 'create_', 'make_', 'build_'):
        factory_names.append(prefix + snake)
    # Also try dropping trailing "V6" / "V36" / "Agent" suffixes
    for suffix in ('V6', 'V32', 'V36', 'V7', 'Agent'):
        if cls_name.endswith(suffix):
            stem = cls_name[: -len(suffix)]
            for prefix in ('get_', 'create_'):
                factory_names.append(prefix + _snake(stem))

    # Append any externally-provided factory names (e.g. AST-scanned from agent module)
    if extra_factories:
        factory_names.extend(extra_factories)

    # For each candidate factory name, check if it's referenced in main_src
    # AND also follow its import alias (e.g. `import get_foo as _bar` → add `_bar`)
    for fn in factory_names:
        if re.search(rf'\b{re.escape(fn)}\s*\(', main_src):
            names.add(fn)
        # Find aliased imports of this factory
        for m in re.finditer(
            rf'from\s+{re.escape(module_path)}\s+import\s+([^\n#]+)', main_src
        ):
            for item in m.group(1).split(','):
                item = item.strip()
                parts = re.split(r'\s+as\s+', item)
                if len(parts) == 2 and parts[0].strip() == fn:
                    names.add(parts[1].strip())
                elif parts[0].strip() == fn:
                    names.add(fn)

    return sorted(names)


def _scan_agent_module_factories(file_path: str) -> List[str]:
    """Open the agent file and find all top-level `def get_*` / `create_*` functions.

    This catches factories whose names don't follow predictable snake_case of the class
    (e.g., class OnChainGraphAlphaAgent → factory get_onchain_graph_agent, not
    get_on_chain_graph_alpha_agent).
    """
    try:
        src = (ROOT / file_path).read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(src)
    except Exception:
        return []
    out = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            n = node.name
            if n.startswith(('get_', 'create_', 'make_', 'build_')) and not n.startswith('_'):
                out.append(n)
    return out


def check_class_wiring(main_src: str, cls_name: str, file_path: str) -> Dict[str, Any]:
    # Normalize multi-line `from X import (...)` before all regex passes
    main_src = _normalize_imports(main_src)
    module_path = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')

    # L1: imported (direct or via agents package, incl. factory fn)
    snake = _smart_snake(cls_name)
    factory_stems = [snake]
    for suffix in ('_agent', '_v6', '_v32', '_v36', '_v7'):
        if snake.endswith(suffix):
            factory_stems.append(snake[: -len(suffix)])
    factory_names = [p + s for p in ('get_', 'create_', 'make_', 'build_') for s in factory_stems]
    # Also include all actual factories defined in the agent module (AST-scanned)
    factory_names += _scan_agent_module_factories(file_path)

    import_patterns = [
        rf'from\s+{re.escape(module_path)}\s+import\s+[^\n#]*\b{re.escape(cls_name)}\b',
        rf'import\s+{re.escape(module_path)}\b',
        rf'from\s+agents\b.*\bimport\b[^\n#]*\b{re.escape(cls_name)}\b',
    ]
    # Also L1-pass if a factory function from this module is imported
    for fn in factory_names:
        import_patterns.append(rf'from\s+{re.escape(module_path)}\s+import\s+[^\n#]*\b{re.escape(fn)}\b')
    l1 = any(re.search(p, main_src) for p in import_patterns)

    # Find all aliases / factory names that represent this class
    names = _discover_aliases(
        main_src, cls_name, module_path,
        extra_factories=_scan_agent_module_factories(file_path),
    )

    # L2: instantiated (via any alias)
    l2 = False
    for n in names:
        if re.search(rf'\b{re.escape(n)}\s*\(', main_src):
            l2 = True
            break

    # L3: attribute set (via any alias)
    l3_matches: List[str] = []
    for n in names:
        l3_matches += re.findall(rf'self\.(\w+)\s*=\s*{re.escape(n)}\s*\(', main_src)
    l3 = bool(l3_matches)

    # L4: method called on instance attribute
    called_methods: Set[str] = set()
    for attr_name in set(l3_matches):
        call_pattern = rf'self\.{re.escape(attr_name)}\.(\w+)\s*\('
        for m in re.finditer(call_pattern, main_src):
            called_methods.add(m.group(1))
    l4 = bool(called_methods)

    if l1 and l2 and l3 and l4:
        verdict = 'ACTIVE'
    elif l1 and l2 and l3:
        verdict = 'INSTANTIATED_BUT_UNUSED'
    elif l1 and l2:
        verdict = 'SOMETHING_CREATED'
    elif l1:
        verdict = 'IMPORTED_ONLY'
    else:
        verdict = 'DEAD'

    return {
        'L1_imported': l1,
        'L2_instantiated': l2,
        'L3_attribute_set': l3,
        'L4_method_called': l4,
        'called_methods': sorted(called_methods),
        'aliases': names,
        'verdict': verdict,
    }


def main():
    print()
    print("=" * 100)
    print("  AGENT WIRING TRUTH  |  which agent files are imported and used in main?")
    print("=" * 100)
    print()

    classes = enumerate_agent_classes()
    print(f"Found {len(classes)} classes in agents/ (AST-enumerated)")
    print()

    main_py = find_main_entry()
    if main_py is None:
        print("[ERR] No main.py found")
        sys.exit(1)

    print(f"Entry: {main_py.relative_to(ROOT)}")
    main_src = main_py.read_text(encoding='utf-8', errors='ignore')

    for extra in ['v36/integration_v36.py', 'integration/integration_v36.py']:
        ep = ROOT / extra
        if ep.exists():
            main_src += '\n' + ep.read_text(encoding='utf-8', errors='ignore')
            print(f"  + included: {extra}")
    print(f"Hot-path source total: {len(main_src):,} chars")
    print()

    print("-" * 110)
    print(f"  {'Class':<38} {'File':<40} {'L1':^4} {'L2':^4} {'L3':^4} {'L4':^4}  {'Verdict':<25}")
    print("-" * 110)

    results = []
    for c in classes:
        if c.get('class') == '<PARSE_FAILED>':
            print(f"  {'<PARSE FAIL>':<38} {c['file']:<40}  --  --  --  --  {c.get('error','')[:25]}")
            continue
        check = check_class_wiring(main_src, c['class'], c['file'])
        results.append({**c, **check})
        print(f"  {c['class']:<38} {c['file']:<40} "
              f"{'Y' if check['L1_imported'] else '.':^4} "
              f"{'Y' if check['L2_instantiated'] else '.':^4} "
              f"{'Y' if check['L3_attribute_set'] else '.':^4} "
              f"{'Y' if check['L4_method_called'] else '.':^4}  "
              f"{check['verdict']:<25}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    from collections import Counter
    verdicts = Counter(r['verdict'] for r in results)
    total = len(results)
    for v in ['ACTIVE', 'INSTANTIATED_BUT_UNUSED', 'SOMETHING_CREATED', 'IMPORTED_ONLY', 'DEAD']:
        count = verdicts.get(v, 0)
        if total > 0:
            pct = 100 * count / total
            print(f"  {v:<30} {count:>3} ({pct:>5.1f}%)")

    print()
    print("=" * 100)
    print("UNWIRED AGENTS")
    print("=" * 100)

    unwired = [r for r in results if r['verdict'] in ('DEAD', 'IMPORTED_ONLY')]
    if not unwired:
        print("  (none — all agents are at least instantiated)")
    else:
        for r in unwired:
            print(f"  - {r['class']:<38} in {r['file']:<40} [{r['verdict']}]")

    partial = [r for r in results if r['verdict'] in ('SOMETHING_CREATED', 'INSTANTIATED_BUT_UNUSED')]
    if partial:
        print()
        print("SEMI-WIRED (created but methods never called):")
        for r in partial:
            print(f"  ~ {r['class']:<38} in {r['file']:<40} [{r['verdict']}]")

    print()
    dead_count = verdicts.get('DEAD', 0) + verdicts.get('IMPORTED_ONLY', 0)
    if dead_count > 0:
        print(f"[WARN] {dead_count} agents dead/unwired — real bugs, not audit noise.")
    else:
        print("[OK] All agents have at least basic wiring.")


if __name__ == '__main__':
    main()
