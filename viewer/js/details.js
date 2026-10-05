import { esc, pretty, getPlanSteps } from './utils.js';

export function setupDetails(cy, LOG) {
    const $details = document.getElementById('details');

    function renderDetails(n) {
        const d = n.data();
        // Enlève de d.id ""::msg_main" s'il est présent:
        const full = LOG.nodes[d.id.replace(/::msg_main$/, '')] || {};
        const when = full.timestamp ? new Date(full.timestamp * 1000).toISOString() : '—';
        const type = d.t;

        const jPrompt = pretty(full.prompt_messages ?? null);
        const jResp = pretty(full.raw_response ?? null);
        const jPlan = pretty(full.plan_snapshot ?? null);
        const jToolIn = pretty(full.tool_input ?? null);
        const jToolOut = pretty(full.tool_output ?? null);
        const jTags = pretty(full.tags ?? null);
        const jUsage = pretty(full.token_usage ?? null);
        const steps = getPlanSteps(full.plan_snapshot);

        $details.innerHTML = `
    <div class="kv">
        <div>ID</div><div class="small">${esc(d.id)}</div>
        <div>Type</div><div>${esc(type)}</div>
        <div>Phase</div><div>${esc(full.phase ?? "")}</div>
        <div>Turn</div><div>${esc(full.turn ?? "")}</div>
        <div>Horodatage</div><div class="small">${esc(when)}</div>
        <div>Erreur</div><div>${full.error ? `<span class="error-badge">${esc(full.error)}</span>` : '—'}</div>
        <div>Durée (s)</div><div>${esc(full.duration_s ?? '—')}</div>
    </div>
    <div class="sep"></div>
    <h3>prompt_messages</h3>
    <andypf-json-viewer id="prompt_messages_json" expanded="1" theme="default-dark" show-data-types="false" show-toolbar="false" expand-icon-type="square" show-copy="false" show-size="false"></andypf-json-viewer>

    ${
            type === 'plan'
            ? `<h3>Étapes du plan</h3>
            ${steps.length
                ? `<ol class="plan-list">${steps.map(s => `<li>${esc(s)}</li>`).join('')}</ol>`
                : `<div class="small muted">Aucune étape trouvée dans <code>plan_snapshot.plan_steps</code>.</div>`}
            <details>
            <summary class="small">Voir le JSON brut du plan</summary>
            <andypf-json-viewer id="plan_snapshot_json" expanded="1" theme="default-dark" show-data-types="false" show-toolbar="false" expand-icon-type="square" show-copy="false" show-size="false"></andypf-json-viewer>
            </details>`
            : ''
        }
    ${
            type === 'tool'
            ? `<h3>Tool</h3>
            <div class="kv"><div>tool_name</div><div>${esc(full.tool_name ?? '—')}</div></div>
            <h3>tool_input</h3><pre class="code">${esc(jToolIn)}</pre>
            <h3>tool_output</h3><pre class="code">${esc(jToolOut)}</pre>`
            : ''
        }

    <h3>raw_response</h3>
    <andypf-json-viewer id="raw_response_json" expanded="1" theme="default-dark" show-data-types="false" show-toolbar="false" expand-icon-type="square" show-copy="false" show-size="false"></andypf-json-viewer>

    <div class="sep"></div>
    <h3>tags</h3>
    <pre class="code small">${esc(jTags)}</pre>

    <h3>token_usage</h3>
    <pre class="code small">${esc(jUsage)}</pre>
    `;
        document.getElementById("prompt_messages_json").data = full.prompt_messages;
        if (document.getElementById("plan_snapshot_json")) {
            document.getElementById("plan_snapshot_json").data = full.plan_snapshot;
        }
        if (document.getElementById("raw_response_json")) {
            document.getElementById("raw_response_json").data = full.raw_response;
        }
    }

    cy.on('tap', 'node', evt => {
        const n = evt.target;
        cy.$(':selected').unselect();
        n.select();
        renderDetails(n);
    });

    if (LOG.root_id && cy.$id(LOG.root_id).nonempty()) {
        const r = cy.$id(LOG.root_id);
        r.select();
        cy.center(r);
        renderDetails(r);
    } else {
        cy.fit(undefined, 30);
    }
}
