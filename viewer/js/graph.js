import { FILLS } from './utils.js';

export function initGraph(elements) {
    return cytoscape({
        container: document.getElementById('cy'),
        elements,
        style: [
            {
                selector: 'node',
                style: {
                    'shape': 'round-rectangle',
                    'width': 'label',
                    'height': 'label',
                    'padding': '8px',
                    'background-color': '#1b2030',
                    'border-width': 1,
                    'border-color': '#3a4155',
                    'label': 'data(label)',
                    'font-size': 20,
                    'color': '#e8ebff',
                    'text-wrap': 'wrap',
                    'text-max-width': '260px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                }
            },
            { selector: 'node[t="message_history"]', style: { 'border-color': '#151a24', 'font-size': 15, 'background-color': '#1d202c' } },
            { selector: 'node[t="message_history_main"]', style: { 'border-color': '#232a3a', 'font-size': 15, 'background-color': '#212531' } },
            { selector: 'node[t="plan"]', style: { 'border-color': FILLS.plan } },
            { selector: 'node[t="tool"]', style: { 'border-color': FILLS.tool } },
            { selector: 'node[t="done"]', style: { 'border-color': FILLS.done, 'background-color': '#151a24' } },
            { selector: 'node[error>0]', style: { 'border-color': FILLS.error } },
            {
                selector: 'edge',
                style: {
                    'width': 1.2,
                    'line-color': '#394255',
                    'target-arrow-color': '#394255',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            },
            {
                selector: ':selected',
                style: { 'border-width': 2.5, 'border-color': 'var(--accent)' }
            }
        ],
        //layout: { name: 'dagre', nodeSep: 24, edgeSep: 12, rankSep: 30, rankDir: 'TB', animate: true, fit: true, nodeDimensionsIncludeLabels: true },
        wheelSensitivity: 0.2
    });
}
