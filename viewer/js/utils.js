export const FILLS = {
    plan: 'rgba(76,120,255,0.14)',
    tool: 'rgba(52,195,143,0.14)',
    done: 'rgba(154,160,166,0.16)',
    other: 'rgba(255,176,32,0.16)',
    error: 'rgba(255,92,124,0.4)'
};

export const esc = (s) =>
    String(s).replace(/[&<>"']/g, m =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[m]
    );

export const truncate = (s, n = 180) =>
    (s && s.length > n * 1.1) ? s.slice(0, n - 1) + "…" : (s ?? "");

export const pretty = (obj, indent = 2) => {
    try {
        let stringerized = obj;
        if (typeof stringerized != "string") {
            stringerized = JSON.stringify(obj, null, indent);
        }
        stringerized = stringerized.replace(/\\n/g, "\n")
            .replace(/\t/g, "  ")
            .replace(/\'/g, "'")
            .replace(/\"/g, '"');
        return stringerized;
    } catch { return String(obj); }
};

export const getPlanSteps = (plan) => {
    if (!plan) return [];
    const arr = Array.isArray(plan.plan_steps) ? plan.plan_steps : [];
    let nb_pending = 0;
    return arr.map(s => {
        if (!s) return null;
        if (typeof s === 'string') return s;
        if (s.status === 'done') return "✅" + (s.descr ?? null);
        nb_pending++;
        return (nb_pending === 1 ? "🔄" : "🔲") + (s.descr ?? null);
    }).filter(Boolean);
};

export const isPlanPhase = (ph) =>
    typeof ph === "string" && ph.startsWith("plan/");

export const nodeType = (n) =>
    n.tool_name ? "tool" :
        (isPlanPhase(n.phase) ? "plan" :
            (["done", "start"].includes(n.phase) ? "done" : "other"));
