# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:dc0606f7-3c77-4e30-97e8-ed1e8eac458f

"""Readable HTML for the election report's EDOM.

The pinned EDT HTML writer renders only ``title``/``heading`` nodes as headings and
prints every other node as a paragraph, so this report's ``heading2``/``heading3``
nodes and pipe tables came out as raw text. This renderer covers exactly the node
shapes ``publish.build_edom`` emits: titles, numbered headings, paragraphs, Markdown
pipe tables and ``- `` bullet lists. Every piece of text is HTML-escaped; no markup
from a source file is passed through.
"""

from __future__ import annotations

from html import escape

STYLE = """
:root { color-scheme: light dark; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --band: #f6f8fa; --bg: #ffffff; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --band: #151b23; --bg: #0d1117; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 12px; }
h2 { font-size: 1.4rem; margin: 40px 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.1rem; margin: 28px 0 6px; }
p, ul { margin: 0 0 12px; }
.muted, h2 ~ p.edt-note { color: var(--muted); }
.table-wrap { overflow-x: auto; margin: 0 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: var(--band); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
"""


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells, current, escaped = [], [], False
    for char in body:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def _is_table(lines: list[str]) -> bool:
    return (len(lines) >= 2 and lines[0].lstrip().startswith("|")
            and set(lines[1].replace("|", "").replace(":", "").replace(" ", "")) <= {"-"}
            and "-" in lines[1])


def _numeric(value: str) -> bool:
    text = value.replace(",", "").replace("%", "").replace("+", "").replace("−", "-").lstrip("<>")
    try:
        float(text)
    except ValueError:
        return False
    return True


def _table(lines: list[str]) -> str:
    header = _cells(lines[0])
    rows = [_cells(line) for line in lines[2:] if line.strip()]
    numeric = [bool(rows) and all(_numeric(row[i]) for row in rows if i < len(row) and row[i] not in {"", "—"})
               for i in range(len(header))]
    out = ['<div class="table-wrap"><table>', "<thead><tr>"]
    out += [f'<th{" class=\"num\"" if numeric[i] else ""} scope="col">{escape(cell)}</th>' for i, cell in enumerate(header)]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(
            f'<td{" class=\"num\"" if i < len(numeric) and numeric[i] else ""}>{escape(cell)}</td>'
            for i, cell in enumerate(row)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _paragraph(node_id: str, text: str) -> str:
    lines = text.splitlines()
    if _is_table(lines):
        return f'<section id="{escape(node_id, quote=True)}">{_table(lines)}</section>'
    if lines and all(line.startswith("- ") for line in lines):
        items = "".join(f"<li>{escape(line[2:])}</li>" for line in lines)
        return f'<ul id="{escape(node_id, quote=True)}">{items}</ul>'
    return f'<p id="{escape(node_id, quote=True)}">' + "<br>".join(escape(line) for line in lines) + "</p>"


def render_node(node: dict) -> str:
    kind = str(node.get("kind", ""))
    text = str(node.get("text", ""))
    node_id = str(node.get("id", ""))
    if kind == "document":
        return "\n".join(render_node(child) for child in node.get("children", []) if isinstance(child, dict))
    if kind == "title":
        return f'<h1 id="{escape(node_id, quote=True)}">{escape(text)}</h1>'
    if kind.startswith("heading"):
        level = node.get("metadata", {}).get("level") or (int(kind[7:]) if kind[7:].isdigit() else 2)
        level = max(2, min(6, int(level)))
        return f'<h{level} id="{escape(node_id, quote=True)}">{escape(text)}</h{level}>'
    return _paragraph(node_id, text)


def edom_to_html(edom: dict, title: str) -> str:
    root = edom.get("root", edom)
    return "\n".join([
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f"<style>{STYLE}</style>",
        "</head>",
        "<body>",
        '<main role="main">',
        render_node(root),
        "</main>",
        "</body>",
        "</html>",
        "",
    ])
