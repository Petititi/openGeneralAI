const form = document.getElementById('askForm');
const input = document.getElementById('question');
const btn = document.getElementById('sendBtn');
const conversation = document.getElementById('conversation');
const status = document.getElementById('status');

async function askServer(q) {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: q })
  });
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.answer || 'Server error');
  return data.answer;
}

// Question and answer are inserted as text (never as HTML): the answer comes from the LLM
function addMessage(boxClass, textClass, text) {
  const card = document.createElement('div');
  card.className = `card ${boxClass}`;
  const body = document.createElement('div');
  body.className = 'card-body';
  const content = document.createElement('div');
  content.className = textClass;
  content.textContent = text;
  body.appendChild(content);
  card.appendChild(body);
  conversation.appendChild(card);
}

function addToConversation(question, answer) {
  addMessage('question-box', 'question', question);
  addMessage('answer-box', 'answer', answer);
  // Scroll to bottom
  conversation.scrollTop = conversation.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;

  btn.disabled = true; 
  input.disabled = true;
  status.textContent = 'Sending...';

  try {
    const a = await askServer(q);
    addToConversation(q, a);
    input.value = '';
    status.textContent = '';
  } catch (err) {
    status.textContent = err.message || String(err);
    // Add error to conversation
    addToConversation(q, `Error: ${err.message || String(err)}`);
  } finally {
    btn.disabled = false; 
    input.disabled = false;
    input.focus();
  }
});
