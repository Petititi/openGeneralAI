import { buildElements } from './data.js';
import { initGraph } from './graph.js';
import { loadTrace, showLoadError } from './load.js';

let LOG;
try {
    LOG = await loadTrace();
} catch (err) {
    showLoadError(err);
    throw err;
}

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
setTimeout(() => {
    cy.layout(grid_layout).run();
    cy.one('layoutstop', () => {
    cy.animate({
        zoom: 1,            // niveau de zoom par défaut
        pan: { x: 0, y: 50}, // centre par défaut
        duration: 500,      // durée en ms
        easing: 'ease-in-out'
    });
    });
}, 2000);

let zoom_toggle = true;
    
document.getElementById('resetBtn').addEventListener('click', () => {
    zoom_toggle = !zoom_toggle;
    if (zoom_toggle) {
        cy.animate({
            zoom: 1,            // niveau de zoom par défaut
            pan: { x: 0, y: 50}, // centre par défaut
            duration: 500,      // durée en ms
            easing: 'ease-in-out'
        });
    }
    else
    {
        cy.layout(grid_layout).run();
    }
});
