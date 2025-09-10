import { nodeType, getPlanSteps, truncate, pretty } from './utils.js';

export function buildElements(LOG) {
    const elements = [];
    const nodes = LOG.nodes || {};
    const children = LOG.children || {};

    let col = 0;
    
    const msgGlobal = `shain_of_thought::msg_global`;
    elements.push({
        data: {
            id: msgGlobal, label:"", phase: "", turn: "", t: "message_history_main",
            error: 0, timestamp: 0,
        }
    });

    for (const [id, n] of Object.entries(nodes)) {
        let row = 0;
        const t = nodeType(n);
        let label = "";
        let tool_output = "";
        let tool_error = "";
        let raw_response = "";
        if (t === "tool") {
            const out = (typeof n.tool_input === "object")
                ? pretty(n.tool_input, 0)
                : (n.tool_input ?? "");
            tool_output = (typeof n.tool_output === "object")
                ? pretty(n.tool_output, 0)
                : (n.tool_output ?? "");
            tool_error = n.error ? `⚠️ ${n.error}` : "";
            label = `🔧 ${n.tool_name || "tool"}\n\n${truncate(out, 220)}`;
        } else if (t === "plan" && n.plan_snapshot) {
            const steps = getPlanSteps(n.plan_snapshot);
            const bullets = steps.map((d, i) => `${i + 1}. ${d}`).join('\n');
            label = `🧭 Plan update \n\n${truncate(bullets, 220)}`;
            raw_response = pretty(n.raw_response);
        } else {
            label = `${n.phase || "?"} (t${n.turn ?? "?"})`;
        }

        const msgMain = `${id}::msg_main`;
        elements.push({
            data: {
                id: msgMain, label, phase: n.phase, turn: n.turn, t,
                error: n.error ? 1 : 0, timestamp: n.timestamp,
                parent: msgGlobal,
                row: row, col: col
            }
        });
        if (Array.isArray(n.prompt_messages)) {
            row = -n.prompt_messages.length;
        }

        // Ajout du chaînage des prompt_messages
        let prevMsgId = null;
        if (Array.isArray(n.prompt_messages) && n.prompt_messages.length > 0) {
            n.prompt_messages.forEach((msg, idx) => {
                const msgId = `${id}::msg${idx}`;
                const msgLabel = truncate(msg.content || "", 220);

                // Créer le noeud pour ce message
                elements.push({
                    data: {
                        id: msgId,
                        label: msgLabel,
                        role: msg.role,
                        type: "prompt_message",
                        t: "message_history",
                        //parent: msgGlobal,
                        row: row, col: col
                    }
                });

                // Relier entre eux les messages successifs
                if (prevMsgId) {
                    elements.push({
                        data: { id: `${prevMsgId}->${msgId}`, source: prevMsgId, target: msgId }
                    });
                }
                prevMsgId = msgId;
                row++;
            });
        }

        if (prevMsgId) {
            elements.push({
                data: { id: `${prevMsgId}->${msgMain}`, source: prevMsgId, target: msgMain }
            });
        }
        if (t === "tool" || t === "plan") {
            const toolResponse = `${id}::msg_response`;
            if (tool_output == null || tool_output === "" || tool_output === "null") {
                tool_output = tool_error;
            }
            let label_response = (t === "tool") ? tool_output : raw_response;
            elements.push({
                data: {
                    id: toolResponse,
                    label: truncate(label_response, 220),
                    role: t,
                    type: "prompt_response",
                    t: "message_history",
                    //parent: msgGlobal,
                    row: row + 1, col: col
                }
            });
            elements.push({
                data: { id: `${toolResponse}->${msgMain}`, source: toolResponse, target: msgMain }
            });
        }
        col++;
    }


    for (const [pid, kids] of Object.entries(children)) {
        (kids || []).forEach(cid => {
            if (nodes[pid] && nodes[cid]) {
                elements.push({ data: { id: `${pid}::msg_main` + "->" + `${cid}::msg_main`, source: `${pid}::msg_main`, target: `${cid}::msg_main` } });
            }
        });
    }

    return elements;
}
