"""Static checks on the LaTeX project.

No compiler is available locally, so this catches the failures that do not need one:
undefined citation/glossary/reference keys, unbalanced braces, missing \\input files and
missing graphics. It cannot replace a real compile.
"""
import re
from pathlib import Path

ROOT = Path(r'C:\Users\HUAWEI\Documents\POCUS-Project\_docs\report\latex')
tex = sorted(ROOT.rglob('*.tex'))
src = {p: p.read_text(encoding='utf8') for p in tex}
allsrc = '\n'.join(src.values())

problems = []


def strip_comments(s):
    return re.sub(r'(?<!\\)%.*', '', s)


# ---------------------------------------------------------------- brace balance --------
for p, s in src.items():
    body = strip_comments(s)
    depth = 0
    for ch in re.sub(r'\\[{}]', '', body):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if depth < 0:
            break
    if depth != 0:
        problems.append(f'{p.name}: brace imbalance ({depth:+d})')

# ---------------------------------------------------------------- \input targets -------
for p, s in src.items():
    for m in re.finditer(r'^[^%]*\\input\{([^}]+)\}', strip_comments(s), re.M):
        target = m.group(1)
        if not (ROOT / target).with_suffix('.tex').exists() and not (ROOT / target).exists():
            problems.append(f'{p.name}: \\input{{{target}}} -> file not found')

# ---------------------------------------------------------------- citations ------------
bib = (ROOT / 'bibfile.bib').read_text(encoding='utf8')
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
cited = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', strip_comments(allsrc)):
    cited |= {k.strip() for k in m.group(1).split(',')}
for k in sorted(cited - bibkeys):
    problems.append(f'\\cite{{{k}}} has no entry in bibfile.bib')
unused = sorted(bibkeys - cited)

# ---------------------------------------------------------------- glossary -------------
gloss = (ROOT / 'glossary.tex').read_text(encoding='utf8')
# Two declaration forms, both {key}{printed form}{description}:
#   \newabbr for acronyms (front matter), \newterm for technical terms (back matter).
entries = dict(re.findall(r'\\new(?:abbr|term)\{([^}]+)\}\{([^}]+)\}', gloss))
glskeys = set(entries)
used = set(re.findall(r'\\gls(?:pl)?\{([^}]+)\}', strip_comments(allsrc)))
used = {k for k in used if '#' not in k}   # \glspl is defined as \gls{#1}s in glossary.tex
for k in sorted(used - glskeys):
    problems.append(f'\\gls{{{k}}} is not defined in glossary.tex')
# \glspl on a name ending in "s" or a hyphenated term produces a wrong plural.
for m in re.finditer(r'\\glspl\{([^}]+)\}', strip_comments(allsrc)):
    key = m.group(1)
    base = entries.get(key)
    if base and base.endswith('s'):
        problems.append(f'\\glspl{{{key}}} -> "{base}s" (bad plural)')

# ---------------------------------------------------------------- refs and labels ------
labels = set(re.findall(r'\\label\{([^}]+)\}', allsrc))
for m in re.finditer(r'\\(?:page)?ref\{([^}]+)\}', strip_comments(allsrc)):
    if m.group(1) not in labels and m.group(1) != 'LastPage':
        problems.append(f'\\ref{{{m.group(1)}}} has no matching \\label')

# ---------------------------------------------------------------- graphics -------------
searchdirs = [ROOT, ROOT / 'figures', ROOT / 'logos']
for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', strip_comments(allsrc)):
    f = m.group(1)
    if not any((d / f).exists() for d in searchdirs):
        problems.append(f'\\includegraphics{{{f}}} -> not found in figures/ or logos/')

# ---------------------------------------------------------------- report ---------------
print('files:', ', '.join(p.name for p in tex))
print(f'citations used: {len(cited)}   bib entries: {len(bibkeys)}   '
      f'glossary entries: {len(glskeys)} ({len(used)} used in text)')
if unused:
    print('bib entries not yet cited (fine, they are for later chapters):',
          ', '.join(unused))
print()
if problems:
    print(f'{len(problems)} PROBLEM(S):')
    for x in problems:
        print('  -', x)
else:
    print('no problems found')
