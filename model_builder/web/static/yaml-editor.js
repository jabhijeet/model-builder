
function buildYaml(filePath, targetCol, taskType) {
  return [
    'nodes:',
    '  - id: ingest_files',
    '    type: task',
    '    plugin: connectors.file',
    '    config:',
    '      paths: ["' + filePath + '"]',
    '      target_col: "' + targetCol + '"',
    '',
    '  - id: validate',
    '    type: task',
    '    plugin: validators.schema',
    '    depends_on: [ingest_files]',
    '',
    '  - id: review_data',
    '    type: gate',
    '    depends_on: [validate]',
    '    message: "Review data profile before training"',
    '',
    '  - id: profile',
    '    type: task',
    '    plugin: core.profile',
    '    depends_on: [review_data]',
    '',
    '  - id: rank_algos',
    '    type: task',
    '    plugin: core.automl_ranker',
    '    depends_on: [profile]',
    '',
    '  - id: select_algos',
    '    type: gate',
    '    depends_on: [rank_algos]',
    '    message: "Select algorithms to train"',
    '',
    '  - id: export_model',
    '    type: task',
    '    plugin: core.export',
    '    depends_on: [select_algos]',
    '',
    '  - id: gen_deploy_instructions',
    '    type: task',
    '    plugin: core.deploy_advisor',
    '    depends_on: [export_model]',
  ].join('\n') + '\n';
}

function syncFormToYaml() {
  var fileEl = document.getElementById('cfg-file');
  var targetEl = document.getElementById('cfg-target');
  var taskEl = document.querySelector('.task-chip.selected');
  if (!fileEl || !targetEl) return;
  var filePath = 'data/raw/' + fileEl.value;
  var targetCol = targetEl.value;
  var taskType = taskEl ? taskEl.dataset.value : 'classification';
  var textarea = document.getElementById('yaml-editor');
  if (textarea) textarea.value = buildYaml(filePath, targetCol, taskType);
}

async function loadColumns(filename) {
  if (!filename) return;
  var resp = await fetch('/api/file-info/' + encodeURIComponent(filename));
  if (!resp.ok) return;
  var data = await resp.json();
  var targetEl = document.getElementById('cfg-target');
  if (!targetEl) return;
  targetEl.innerHTML = '';
  data.columns.forEach(function (col) {
    var opt = document.createElement('option');
    opt.value = col.name;
    opt.textContent = col.name + ' (' + col.dtype + ')';
    if (col.name === data.detected_target) opt.selected = true;
    targetEl.appendChild(opt);
  });
  var infoEl = document.getElementById('file-info');
  if (infoEl) infoEl.textContent = data.rows + ' rows · ' + data.columns.length + ' cols';
  syncFormToYaml();
}

async function saveYaml() {
  var textarea = document.getElementById('yaml-editor');
  var btn = document.getElementById('save-yaml-btn');
  if (!textarea || !btn) return;
  var content = textarea.value;
  var resp = await fetch('/api/yaml', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: content,
  });
  if (resp.ok) {
    btn.textContent = 'Saved!';
    setTimeout(function () { btn.textContent = 'Save'; }, 1500);
  } else {
    btn.textContent = 'Error';
    setTimeout(function () { btn.textContent = 'Save'; }, 1500);
  }
}

async function validateYaml() {
  var textarea = document.getElementById('yaml-editor');
  if (!textarea) return;
  var content = textarea.value;
  var resp = await fetch('/api/yaml/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: content,
  });
  var data = await resp.json();
  var errEl = document.getElementById('yaml-error');
  if (!errEl) return;
  if (data.valid) {
    errEl.hidden = true;
  } else {
    errEl.textContent = data.error;
    errEl.hidden = false;
  }
}

function toggleChip(el, groupClass) {
  el.classList.toggle('selected');
  syncFormToYaml();
}

function toggleTaskChip(el) {
  document.querySelectorAll('.task-chip').forEach(function (c) { c.classList.remove('selected'); });
  el.classList.add('selected');
  syncFormToYaml();
}

document.addEventListener('DOMContentLoaded', function () {
  var fileEl = document.getElementById('cfg-file');
  var targetEl = document.getElementById('cfg-target');
  if (fileEl && fileEl.value) loadColumns(fileEl.value);
  if (fileEl) fileEl.addEventListener('change', function () { loadColumns(fileEl.value); });
  if (targetEl) targetEl.addEventListener('change', syncFormToYaml);
});
