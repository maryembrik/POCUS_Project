"""Static checks on the LaTeX project.

No compiler is available locally, so this catches the failures that do not need one:
undefined citation/glossary/reference keys, unbalanced braces, missing \\input files and
missing graphics. It cannot replace a real compile.
"""
import re
from pathlib import Path

# The directory this file sits in, NOT a hardcoded path. It was hardcoded, which meant the
# checker always read the working copy no matter where it was run from -- so running it inside
# an extracted zip silently checked the original sources instead, and reported that the zip was
# complete without having opened a single file in it. A check that cannot be pointed at a
# different copy cannot verify a copy.
ROOT = Path(__file__).resolve().parent
tex = sorted(ROOT.rglob('*.tex'))
src = {p: p.read_text(encoding='utf8') for p in tex}
allsrc = '\n'.join(src.values())

problems = []
notes = []


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
# An \includegraphics inside \IfFileExists is a BRANCH: exactly one of the alternatives is
# compiled, so reporting every missing one is noise. The title page offers .png and .jpg for
# each logo, which produced four "problems" for two images that are simply optional -- and a
# checker that cries wolf stops being read, which costs more than the check is worth.
searchdirs = [ROOT, ROOT / 'figures', ROOT / 'logos']
# Not `src`: that name holds the {path: text} mapping the later checks read from, and
# rebinding it here silently broke them.
flat = strip_comments(allsrc)

guarded: set[str] = set()
for m in re.finditer(r'\\IfFileExists\{([^}]+)\}', flat):
    guarded.add(Path(m.group(1)).name)

for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', flat):
    f = m.group(1)
    if any((d / f).exists() for d in searchdirs):
        continue
    if Path(f).name in guarded:
        continue          # a guarded alternative; its absence is handled in the document
    problems.append(f'\\includegraphics{{{f}}} -> not found in figures/ or logos/')

# Report the guarded images once, as information rather than as a problem: the document
# compiles without them, but the title page will carry a red placeholder where a logo goes.
for stem in sorted({Path(g).stem for g in guarded}):
    if not any((d / f'{stem}{ext}').exists()
               for d in searchdirs for ext in ('.png', '.jpg', '.jpeg', '.pdf')):
        notes.append(f'optional image {stem}.* not present '
                     f'(the document still compiles; the slot shows a placeholder)')

# ---------------------------------------------------------------- table rows ------------
# The first real compile died on rows ending in ONE backslash instead of two -- a row break
# that an editing script had eaten. TeX reports it far from the cause ("Misplaced \\noalign"),
# and the brace check cannot see it because the braces are balanced. Cheap to detect here.
BS = chr(92)
for p, text in src.items():
    depth = 0
    for i, line in enumerate(text.split('\n'), 1):
        if re.search(r'\\begin\{tabular', line):
            depth += 1
        if re.search(r'\\end\{tabular', line):
            depth -= 1
        if depth <= 0:
            continue
        body = strip_comments(line).rstrip()
        if not body or body.endswith(BS * 2):
            continue
        n = len(body) - len(body.rstrip(BS))
        if n == 1:
            problems.append(f'{p.name}:{i}: table row ends in one backslash, '
                            f'expected two ({body.strip()[:46]}...)')

# ---------------------------------------------------------------- tikz key clashes ------
# A style named after a built-in TikZ key is read as that key. `out/.style` made the whole
# reasoning figure fail with "The key '/tikz/out' requires a value", which names the symptom
# and not the cause. These are the built-ins a diagram is most likely to shadow by accident.
RESERVED = {'in', 'out', 'at', 'above', 'below', 'left', 'right', 'anchor', 'name', 'scale',
            'shift', 'rotate', 'label', 'pos', 'draw', 'fill', 'text', 'node', 'to', 'edge',
            'gate', 'sloped', 'midway', 'near start', 'near end'}
for p, text in src.items():
    for m in re.finditer(r'(\w[\w ]*)/\.style\s*=', strip_comments(text)):
        name = m.group(1).strip()
        if name in RESERVED:
            problems.append(f'{p.name}: tikz style "{name}" shadows a built-in TikZ key '
                            f'-- rename it (e.g. "{name}box")')

# ---------------------------------------------------------------- sectioning depth -----
# \chapter does not exist in the article class. An appendix heading was written with it, and
# because an undefined control sequence stops the compile rather than degrading, the whole
# document would have failed at that line -- while every check above passed, since the braces
# balance, the label resolves and nothing is missing. The class is read rather than assumed,
# so this stays correct if the document ever moves to report or book.
cls = re.search(r'\\documentclass(?:\[[^\]]*\])?\{(\w+)\}',
                strip_comments(''.join(t for p, t in src.items() if p.name == 'main.tex')))
if cls and cls.group(1) in ('article', 'proc', 'letter'):
    for p, text in src.items():
        for i, line in enumerate(strip_comments(text).split('\n'), 1):
            if re.match(r'\s*\\chapter\*?\{', line):
                problems.append(f'{p.name}:{i}: \\chapter does not exist in the '
                                f'{cls.group(1)} class -- the compile stops here '
                                f'({line.strip()[:46]})')

# ---------------------------------------------------------------- report ---------------
print('files:', ', '.join(p.name for p in tex))
print(f'citations used: {len(cited)}   bib entries: {len(bibkeys)}   '
      f'glossary entries: {len(glskeys)} ({len(used)} used in text)')
if unused:
    print('bib entries not yet cited (fine, they are for later chapters):',
          ', '.join(unused))
print()
for n in notes:
    print('note:', n)
if notes:
    print()
if problems:
    print(f'{len(problems)} PROBLEM(S):')
    for x in problems:
        print('  -', x)
else:
    print('no problems found')
