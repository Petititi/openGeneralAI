import { buildElements } from './data.js';
import { initGraph } from './graph.js';
import { setupFilters } from './filters.js';
import { setupDetails } from './details.js';

const elements = buildElements(LOG);
const cy = initGraph(elements);
let grid_layout = {
    name: 'grid',
    fit: true,            // Ajuste le graphe pour qu'il tienne dans la vue
    padding: 30,          // Marge autour du graphe
    avoidOverlap: true,   // Empêche le chevauchement des nœuds
    avoidOverlapPadding: 30, // Espace minimal entre nœuds
    animate: true,        // Anime la transition
    condense: true,      // Si true, resserre la grille
    rows: undefined,      // Nombre de lignes (calculé automatiquement si non défini)
    cols: undefined,      // Nombre de colonnes (calculé automatiquement)
    position: function (node) {
        return {
            row: node.data('row'),
            col: node.data('col')
        };
    },
};

let dagre_layout = { name: 'dagre', nodeSep: 24, edgeSep: 12, rankSep: 30, rankDir: 'TB', animate: true, fit: true, nodeDimensionsIncludeLabels: true };

let is_dagre = false;
cy.layout(grid_layout).run();

document.getElementById("count-nodes").textContent =
    elements.filter(e => e.data && !e.data.source).length;
document.getElementById("count-edges").textContent =
    elements.filter(e => e.data && !!e.data.source).length;

setupFilters(cy);
setupDetails(cy, LOG);

document.getElementById('btn-layout').onclick = () =>
{
    is_dagre = !is_dagre;
    if (is_dagre)
        cy.layout(dagre_layout).run();
    else
        cy.layout(grid_layout).run();
}

document.getElementById('btn-fit').onclick = () =>
    cy.fit(undefined, 30);

document.getElementById('btn-export').onclick = () => {
    const blob = new Blob([JSON.stringify(LOG, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `trajectory_{self.run_id}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
};

const resizer = document.querySelector(".resizer");
const aside = document.querySelector("aside");
const app = document.querySelector("#app");

let isResizing = false;

resizer.addEventListener("mousedown", (e) => {
    isResizing = true;
    document.body.style.cursor = "ew-resize";
});

document.addEventListener("mousemove", (e) => {
    if (!isResizing) return;
    const newWidth = app.offsetWidth - e.clientX;
    aside.style.width = newWidth + "px";
});

document.addEventListener("mouseup", () => {
    isResizing = false;
    document.body.style.cursor = "default";
});
