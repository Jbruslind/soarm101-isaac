#!/usr/bin/env bash
# Static checks for the tutorials (no GPU, no Isaac Sim, host python3 only).
#
#   1. Every tutorial script (tutorials/**/*.py) compiles (python3 -m py_compile).
#   2. Every relative Markdown link in docs/tutorials/ and tutorials/ points at a file
#      or directory that exists (#anchors and http(s)/mailto links are not checked).
#   3. Every code excerpt marked with <!-- snippet: path#name --> matches the
#      "# [start:name]" ... "# [end:name]" region of that script, so pages cannot
#      drift from the code they explain.
#
# Usage: ./scripts/check_tutorials.sh        (exit 0 = all good)
# Meant to become a CI step once the repo has CI (#3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

exec python3 - "$@" <<'EOF'
import pathlib
import py_compile
import re
import sys
import tempfile
import textwrap

root = pathlib.Path(".").resolve()
errors = []

# 1. Scripts compile. Byte code goes to a temp dir so no __pycache__ lands in the tree.
scripts = sorted(root.glob("tutorials/**/*.py"))
with tempfile.TemporaryDirectory() as tmp:
    for i, path in enumerate(scripts):
        try:
            py_compile.compile(str(path), cfile=f"{tmp}/{i}.pyc", doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{path.relative_to(root)}: does not compile\n{exc.msg}")

# 2. Relative links resolve.
pages = sorted(root.glob("docs/tutorials/**/*.md")) + sorted(root.glob("tutorials/**/*.md"))
# The template's links are placeholders relative to docs/tutorials/fasteners/, where copies live.
pages = [p for p in pages if p.name != "TEMPLATE.md"]
link_re = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
fence_re = re.compile(r"^```.*?^```", re.S | re.M)
links = 0
for page in pages:
    text = fence_re.sub("", page.read_text())  # links inside code blocks are examples
    text = re.sub(r"`[^`\n]*`", "", text)
    for target in link_re.findall(text):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        links += 1
        rel = target.split("#", 1)[0]
        if not (page.parent / rel).exists():
            errors.append(f"{page.relative_to(root)}: broken link -> {target}")

# 3. Snippets match their script regions.
snip_re = re.compile(r"<!--\s*snippet:\s*(\S+?)#(\S+)\s*-->\s*\n```[a-z]*\n(.*?)^```", re.S | re.M)
snippets = 0
for page in pages:
    for rel, name, body in snip_re.findall(page.read_text()):
        snippets += 1
        src = root / rel
        if not src.exists():
            errors.append(f"{page.relative_to(root)}: snippet source missing: {rel}")
            continue
        m = re.search(rf"^[ \t]*# \[start:{re.escape(name)}\][ \t]*\n(.*?)^[ \t]*# \[end:{re.escape(name)}\]",
                      src.read_text(), re.S | re.M)
        if not m:
            errors.append(f"{page.relative_to(root)}: no region [start:{name}] in {rel}")
        elif textwrap.dedent(m.group(1)).strip() != textwrap.dedent(body).strip():
            errors.append(f"{page.relative_to(root)}: snippet {rel}#{name} is out of date (copy the region again)")

print(f"check_tutorials: {len(scripts)} scripts, {len(pages)} pages, {links} links, {snippets} snippets")
for e in errors:
    print(f"  [FAIL] {e}")
print("check_tutorials: " + ("FAIL" if errors else "OK"))
sys.exit(1 if errors else 0)
EOF
