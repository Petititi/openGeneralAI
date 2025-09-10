export function setupFilters(cy) {
    const phaseSel = document.getElementById('phase-filter');
    const search = document.getElementById('search');

    function applyFilters() {
        const phase = phaseSel.value || "";
        const q = (search.value || "").toLowerCase();
        cy.nodes().forEach(n => {
            const d = n.data();
            const phaseOk = !phase || (phase.endsWith('/') ? String(d.phase || '').startsWith(phase) : d.phase === phase);
            const text = (d.label || '') + ' ' + (d.phase || '') + ' ' + (String(d.turn || '')) + ' ' + d.id;
            const queryOk = !q || text.toLowerCase().includes(q);
            n.style('display', (phaseOk && queryOk) ? 'element' : 'none');
        });
        cy.edges().forEach(e => {
            const visible = e.source().style('display') !== 'none' && e.target().style('display') !== 'none';
            e.style('display', visible ? 'element' : 'none');
        });
    }

    phaseSel.onchange = applyFilters;
    search.oninput = applyFilters;
}
