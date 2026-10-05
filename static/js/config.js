let providers = [];
let modelsByProvider = {};
let cachedConfig = null;
let has_key = {};

async function loadConfig() {
  const res = await fetch('/api/config');
  const data = await res.json();
  if (!data.ok) throw new Error('Impossible de charger la configuration');
  cachedConfig = data.config;
  providers = data.providers;
  modelsByProvider = data.models;
  has_key = data.has_key;

  const providerSel = document.getElementById('provider');
  const modelSel = document.getElementById('model');

  fillOptions(providerSel, providers);
  providerSel.value = data.config.provider;
  updateModels();
  update_api_field(data.config.provider)
  modelSel.value = data.config.model;
}

function update_api_field(choosen_provider)
{
  let field_api = document.getElementById('apiKey');
  let resetKey = document.getElementById('resetKey');
  if (has_key[choosen_provider])
  {
    field_api.setAttribute('disabled', 'disabled');
    resetKey.removeAttribute('hidden');
    field_api.classList.add('opacity-75');
    field_api.classList.add('text-muted');
    field_api.value = '';
    field_api.placeholder = 'Clé API existante...';
  }
  else
  {
    field_api.removeAttribute('disabled');
    resetKey.setAttribute('hidden','hidden');
    field_api.classList.remove('opacity-75');
    field_api.classList.remove('text-muted');
    field_api.value = '';
    field_api.placeholder = 'Entrez votre clé API...';
  }
}

// Build <option> elements as text, never as HTML
function fillOptions(select, values) {
  select.replaceChildren(...values.map(v => new Option(v, v)));
}

function updateModels() {
  const providerSel = document.getElementById('provider');
  const modelSel = document.getElementById('model');
  const modelFilter = document.getElementById('modelFilter');
  const modelCount = document.getElementById('modelCount');
  const provider = providerSel.value;
  const list = modelsByProvider[provider] || [];
  const q = (modelFilter.value || '').toLowerCase();
  const filtered = q ? list.filter(m => m.toLowerCase().includes(q)) : list;
  fillOptions(modelSel, filtered);
  modelCount.textContent = `${filtered.length} / ${list.length}`;
}

document.getElementById('provider').addEventListener('change', () => {
  const choosen_provider = document.getElementById('provider').value
  updateModels();
  update_api_field(choosen_provider);
  const modelSel = document.getElementById('model');
  if (cachedConfig && cachedConfig.provider === choosen_provider) {
    modelSel.value = cachedConfig.model;
  }
});

document.getElementById('resetKey').addEventListener('click', async () => {
  const choosen_provider = document.getElementById('provider').value
  has_key[choosen_provider] = false;
  update_api_field(choosen_provider);
});

document.getElementById('modelFilter').addEventListener('input', updateModels);

document.getElementById('configForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const provider = document.getElementById('provider').value;
  const model = document.getElementById('model').value;
  const apiKey = document.getElementById('apiKey').value;
  const btn = document.getElementById('saveBtn');
  const status = document.getElementById('status');
  btn.disabled = true; status.textContent = 'Enregistrement...';
  let body_content = { provider, model, apiKey };
  if (has_key[provider])
    body_content = { provider, model };
  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body_content)
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Erreur serveur');
    status.textContent = 'Configuration enregistrée';
  } catch (err) {
    status.textContent = err.message || String(err);
  } finally {
    btn.disabled = false;
  }
});

loadConfig().catch(err => {
  document.getElementById('status').textContent = err.message || String(err);
});
