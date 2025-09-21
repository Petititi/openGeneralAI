import { buildElements } from './data.js';
import { initGraph } from './graph.js';

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

cy.layout(grid_layout).run();
