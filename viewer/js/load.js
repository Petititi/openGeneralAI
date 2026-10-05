// Load the trace to display from the JSON file given in the page URL: ?trace=<url>.
// Only same-origin files are accepted: a trace holds prompts and tool outputs.
export async function loadTrace() {
    const param = new URLSearchParams(window.location.search).get('trace');
    if (!param) {
        throw new Error('No trace to display: add ?trace=<path of a trace JSON file> to the URL.');
    }
    const url = new URL(param, window.location.href);
    if (url.origin !== window.location.origin) {
        throw new Error(`Refusing to load a trace from another site: ${url.origin}`);
    }
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`Cannot load ${url.pathname}: HTTP ${res.status}`);
    }
    return res.json();
}

// Show a loading error in the page instead of failing silently in the console.
export function showLoadError(err, container = document.body) {
    const box = document.createElement('div');
    box.className = 'load-error';
    box.textContent = err.message || String(err);
    container.appendChild(box);
}
