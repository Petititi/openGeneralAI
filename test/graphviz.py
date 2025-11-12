import html, math

def write_trace_dot_advanced(trace: dict, path: str = "trace.dot", title: str = "Agent trajectory") -> str:
    nodes = trace["nodes"]
    children = trace.get("children", {})
    def esc(s: str) -> str:
        # échappe pour DOT (suffisant pour notre usage)
        return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    def short(nid: str) -> str:
        return nid.split("-")[0]

    phase_style = {
        "plan/create": {"fill": "#D6EAF8", "shape": "box",         "border": "#3498DB"},
        "plan/update": {"fill": "#AED6F1", "shape": "box",         "border": "#2E86C1"},
        "plan/recover":{"fill": "#FAD7A0", "shape": "octagon",     "border": "#E67E22"},
        "act/run":     {"fill": "#E5E7E9", "shape": "ellipse",     "border": "#7F8C8D"},
        "done":        {"fill": "#D5F5E3", "shape": "doublecircle","border": "#27AE60"},
    }

    # groupe les nœuds par tour
    turns = {}
    for nid, n in nodes.items():
        turns.setdefault(n["turn"], []).append(nid)

    lines = [
        f'digraph "{esc(title)}" {{',
        '  rankdir=LR; labelloc=t; fontsize=22; fontname="Helvetica";',
        f'  label="{esc(title)}";',
        '  node [fontname="Helvetica", style="filled,rounded", fontsize=12];',
        '  edge [color="#95A5A6"];'
    ]

    # sous-graphes par tour
    for turn, nids in sorted(turns.items()):
        lines.append(f'  subgraph cluster_{turn} {{')
        lines.append(f'    label="turn {turn}"; color="#DDDDDD"; style="dashed,rounded";')
        for nid in nids:
            n = nodes[nid]
            sty = phase_style.get(n["phase"], {"fill": "#FFFFFF", "shape": "box", "border": "#333333"})
            tok = (n.get("token_usage") or {}).get("total", 0)
            dur = n.get("duration_s")
            dur_txt = f"{dur:.2f}s" if isinstance(dur, (int, float)) and math.isfinite(dur or 0) else "—"
            status = "✔ done" if n.get("is_done") else ""
            err = n.get("error")
            label = f'{{{short(nid)}|{n["phase"]}\\n⏱ {dur_txt} | 🔤 tok {tok}{" | ⚠" if err else ""}{" | " + status if status else ""}}}'
            tooltip = f'{n["phase"]} — turn {n["turn"]}'
            lines.append(
                f'    "{nid}" [label="{esc(label)}", shape="{sty["shape"]}", fillcolor="{sty["fill"]}", '
                f'color="{sty["border"]}", tooltip="{esc(tooltip)}"];'
            )
        lines.append("  }")

    # arêtes parent → enfant
    for pid, childs in children.items():
        for cid in childs:
            lines.append(f'  "{pid}" -> "{cid}";')

    # Légende
    lines += [
        '  subgraph cluster_legend {',
        '    label="Légende"; color="#EEEEEE"; style="rounded";',
        '    key [shape=none, margin=0, label=<',
        '      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">',
        '        <TR><TD><B>Couleur</B></TD><TD><B>Phase</B></TD></TR>',
    ]
    for ph, sty in phase_style.items():
        lines.append(f'        <TR><TD BGCOLOR="{sty["fill"]}"></TD><TD>{html.escape(ph)}</TD></TR>')
    lines += [
        '      </TABLE>',
        '    >];',
        '  }',
        '}'
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def write_trace_dot_vertical(trace: dict, path: str = "trace.dot",
                             title: str = "Agent — Trajectoire (verticale)",
                             show_rows_by_turn: bool = True) -> str:
    nodes = trace["nodes"]
    children = trace.get("children", {})

    def esc(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    def short(nid: str) -> str:
        return nid.split("-")[0]

    # Couleurs par phase (toutes en rectangles arrondis)
    phase_fill = {
        "plan/create": "#D6EAF8",
        "plan/update": "#AED6F1",
        "plan/recover":"#FAD7A0",
        "act/run":     "#E5E7E9",
        "done":        "#D5F5E3",
    }
    # Regrouper par tour
    turns = {}
    for nid, n in nodes.items():
        turns.setdefault(n["turn"], []).append(nid)

    lines = [
        f'digraph "{esc(title)}" {{',
        '  rankdir=TB;',                       # ← vertical
        '  labelloc=t; fontsize=22; fontname="Helvetica";',
        f'  label="{esc(title)}";',
        '  graph [splines=true, nodesep=0.45, ranksep=0.6];',
        '  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#7F8C8D"];',
        '  edge  [color="#95A5A6", arrowsize=0.7];'
    ]

    # Nœuds
    for nid, n in nodes.items():
        phase = n["phase"]
        fill  = phase_fill.get(phase, "#FFFFFF")
        tok = (n.get("token_usage") or {}).get("total", 0)
        dur = n.get("duration_s")
        dur_txt = f"{dur:.2f}s" if isinstance(dur, (int, float)) and math.isfinite(dur or 0) else "—"
        err = n.get("error")
        is_done = bool(n.get("is_done"))

        # Mettre le NOM DE L’OUTIL en avant pour act/run
        tool_name = ""
        if phase == "act/run":
            tool_name = n.get("tool_name") or (n.get("tags") or {}).get("tool_request", "")
        tool_title = ""
        if tool_name:
            tool_title = f'🛠️ <B>{html.escape(str(tool_name))}</B>' if tool_name else None

        # Label HTML (table) pour mieux contrôler la mise en page
        rows = []
        title_row = tool_title if tool_title else f'<B>{html.escape(phase)}</B>'
        rows.append(f'<TR><TD>{title_row}</TD></TR>')
        rows.append(f'<TR><TD>id: {html.escape(short(nid))}</TD></TR>')
        rows.append(f'<TR><TD>⏱ {dur_txt} • 🔤 {tok} tok</TD></TR>')
        if is_done: rows.append('<TR><TD>✔ done</TD></TR>')
        if err:     rows.append('<TR><TD><FONT COLOR="#C0392B">⚠ '
                                f'{html.escape(str(err))[:80]}{"…" if len(str(err))>80 else ""}</FONT></TD></TR>')
        label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">{"".join(rows)}</TABLE>>'

        extra = []
        if is_done:      extra.append('peripheries=2')       # double bord pour "done"
        if phase=="act/run": extra.append('fontsize=13')     # un peu plus grand pour l’action
        tooltip = n.get("prompt_preview") or f'{phase} (turn {n["turn"]})'

        lines.append(
            f'  "{nid}" [label={label}, fillcolor="{fill}", tooltip="{esc(tooltip)}", {", ".join(extra)}];'
        )

    # Regrouper visuellement par tour (rang = même ligne)
    if show_rows_by_turn:
        for turn, nids in sorted(turns.items()):
            lines.append(f'  subgraph cluster_turn_{turn} {{')
            lines.append('    style="dashed,rounded"; color="#DDDDDD";')
            lines.append(f'    label="tour {turn}";')
            lines.append('    { rank=same;')
            for nid in nids:
                lines.append(f'      "{nid}";')
            lines.append('    }')
            lines.append('  }')

    # Arêtes parent → enfant
    for pid, childs in children.items():
        for cid in childs:
            lines.append(f'  "{pid}" -> "{cid}";')

    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def write_trace_html_viewer(trace: dict, svg_path: str = "trace.svg", out_html: str = "trace_viewer.html"):
    # charge l'SVG produit par Graphviz
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    # on prépare un mapping node_id -> messages (role, content)
    nodes = trace["nodes"]
    data = {}
    for nid, n in nodes.items():
        msgs = n.get("prompt_messages") or []
        # n’affiche que role+content, tu peux enrichir si tu veux
        data[nid] = [{"role": m.get("role"), "content": m.get("content", "")} for m in msgs]

    html_text = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<title>Trace interactive</title>
<style>
  body {{ margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial; }}
  .wrap {{ display: grid; grid-template-columns: 1fr 420px; height: 100vh; }}
  #graph {{ overflow: auto; background:#fafafa; border-right:1px solid #e5e5e5; }}
  #details {{ overflow:auto; padding:16px; }}
  #details h2 {{ margin:0 0 8px 0; font-size:16px; }}
  .msg {{ border:1px solid #eee; border-radius:8px; padding:8px 10px; margin:8px 0; background:#fff; }}
  .role {{ font-weight:600; color:#555; margin-bottom:6px; }}
  .content {{ white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:12px; }}
  .hint {{ color:#888; font-size:12px; margin-top:8px; }}
  .highlight g, .highlight path, .highlight polygon, .highlight ellipse, .highlight rect {{ stroke:#2c7be5 !important; stroke-width:2px !important; }}
</style>
</head>
<body>
<div class="wrap">
  <div id="graph">{svg_text}</div>
  <div id="details">
    <h2>🧭 Sélection</h2>
    <div class="hint">Clique un nœud pour voir ses messages.</div>
  </div>
</div>
<script>
const DATA = {json.dumps(data, ensure_ascii=False)};
const graph = document.getElementById('graph');
const svg = graph.querySelector('svg');

// utilitaires
function clearHighlights() {{
  svg.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
}}
function highlightNode(id) {{
  const el = svg.getElementById(id);
  if (el) el.classList.add('highlight');
}}

// rendu panneau de droite
function renderDetails(id) {{
  const box = document.getElementById('details');
  const msgs = DATA[id] || [];
  const title = `<h2>Node: <code>${{id}}</code> — ${msgs.length} message(s)</h2>`;
  const items = msgs.map(m => `
    <div class="msg">
      <div class="role">${{m.role || ""}}</div>
      <div class="content">${{(m.content || "").replace(/&/g,"&amp;").replace(/</g,"&lt;")}}</div>
    </div>`).join("");
  box.innerHTML = title + items || '<div class="hint">Aucun message</div>';
}}

// branche les clics (nécessite id="{nid}" dans le DOT)
Object.keys(DATA).forEach(id => {{
  const el = svg.getElementById(id);
  if (!el) return;
  el.style.cursor = 'pointer';
  el.addEventListener('click', (ev) => {{
    ev.preventDefault();
    clearHighlights();
    highlightNode(id);
    renderDetails(id);
  }});
}});
</script>
</body>
</html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)
    return out_html

"""
B) Popover “dépliable” dans le graphe (overlay SVG)

Si tu veux réellement “déplier” dans le nœud, on peut ajouter un petit popover SVG qui apparaît/disparaît près du nœud cliqué. C’est un peu plus verbeux, mais voici l’essentiel à intégrer dans le même HTML (remplace la partie <script> précédente) :

<script>
const DATA = /* … même mapping nid -> messages … */;
const graph = document.getElementById('graph');
const svg = graph.querySelector('svg');

// crée un calque pour le popover
const pop = document.createElementNS("http://www.w3.org/2000/svg", "g");
pop.setAttribute("id", "popover");
pop.style.display = "none";
svg.appendChild(pop);

function showPopoverFor(id) {
  const node = svg.getElementById(id);
  if (!node) return;
  // calcule la boîte englobante
  const bb = node.getBBox();
  const x = bb.x + bb.width + 12;   // à droite du nœud
  const y = bb.y;                   // aligné en haut

  const msgs = (DATA[id] || []).slice(0, 8); // limite d’aperçu
  const lines = msgs.map(m => (m.role||"") + ": " + (m.content||"").slice(0,120).replace(/\n/g," "));

  // efface contenu précédent
  while (pop.firstChild) pop.removeChild(pop.firstChild);

  // fond
  const padX=10, padY=10, lineH=16, width=360, height = padY*2 + lineH*(Math.max(1, lines.length)+1);
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", x); rect.setAttribute("y", y);
  rect.setAttribute("rx", 8); rect.setAttribute("ry", 8);
  rect.setAttribute("width", width); rect.setAttribute("height", height);
  rect.setAttribute("fill", "#ffffff"); rect.setAttribute("stroke", "#2c3e50"); rect.setAttribute("stroke-width", "1");
  pop.appendChild(rect);

  // titre
  const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
  title.setAttribute("x", x+padX); title.setAttribute("y", y+padY+12);
  title.setAttribute("font-size", "12"); title.setAttribute("font-weight", "600");
  title.textContent = "Messages (" + (DATA[id]?.length||0) + ")";
  pop.appendChild(title);

  // lignes
  lines.forEach((t, i) => {
    const tx = document.createElementNS("http://www.w3.org/2000/svg", "text");
    tx.setAttribute("x", x+padX); tx.setAttribute("y", y+padY+28 + i*lineH);
    tx.setAttribute("font-size", "12");
    tx.textContent = t;
    pop.appendChild(tx);
  });

  pop.style.display = "block";
}
function hidePopover(){ pop.style.display = "none"; }

Object.keys(DATA).forEach(id => {
  const el = svg.getElementById(id);
  if (!el) return;
  el.style.cursor = 'pointer';
  el.addEventListener('click', (ev) => {
    ev.preventDefault();
    if (pop.style.display !== "none" && pop.dataset.for === id) { hidePopover(); return; }
    pop.dataset.for = id;
    showPopoverFor(id);
  });
});

// clic fond pour fermer
svg.addEventListener('click', (ev) => {
  if (ev.target.closest('#popover')) return; // clic dans le popover
  // si clic sur un node: déjà géré; sinon, on ferme
  if (!ev.target.closest('g.node')) hidePopover();
});
</script>

Bonus : remplace le texte brut par un <foreignObject> pour afficher un mini bloc HTML scrollable si tu veux le contenu complet.
"""