function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function runPrediction() {
  var inputs = document.querySelectorAll('.predict-input');
  var features = {};
  inputs.forEach(function (inp) {
    features[inp.dataset.feature] = inp.type === 'number' ? parseFloat(inp.value || '0') : inp.value;
  });

  var btn = document.getElementById('predict-btn');
  btn.textContent = 'Predicting...';
  btn.disabled = true;

  var resultEl = document.getElementById('predict-result');

  try {
    var resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: features }),
    });

    if (!resp.ok) {
      resultEl.innerHTML = '<div class="alert alert-warn">Prediction failed. Is the model exported?</div>';
      resultEl.hidden = false;
      return;
    }

    var data = await resp.json();
    var confPct = data.confidence !== null ? Math.round(data.confidence * 100) + '%' : '';
    var driverHtml = data.top_feature
      ? '<div class="text-muted small mt-8">Top driver: <strong>' + escHtml(data.top_feature) + '</strong> = ' + escHtml(String(data.top_feature_value)) + '</div>'
      : '';

    resultEl.innerHTML =
      '<div class="predict-result-inner">' +
      '<div class="predict-class">' + escHtml(data.prediction) + '</div>' +
      (confPct ? '<div class="predict-conf">Confidence: ' + confPct + '</div>' : '') +
      driverHtml +
      '</div>';
    resultEl.hidden = false;
  } catch (e) {
    resultEl.innerHTML = '<div class="alert alert-warn">Network error. Check connection.</div>';
    resultEl.hidden = false;
  } finally {
    btn.textContent = 'Predict →';
    btn.disabled = false;
  }
}

function switchQueryTab(tab) {
  document.querySelectorAll('.query-tab').forEach(function (t) { t.classList.remove('active'); });
  document.querySelectorAll('.query-panel').forEach(function (p) { p.hidden = true; });
  document.querySelector('.query-tab[data-tab="' + tab + '"]').classList.add('active');
  document.getElementById('panel-' + tab).hidden = false;
  if (tab === 'explain') loadExplain();
}

var _explainLoaded = false;

async function loadExplain() {
  if (_explainLoaded) return;
  var el = document.getElementById('explain-content');
  var resp = await fetch('/api/explain');
  if (!resp.ok) {
    el.innerHTML = '<p class="text-muted">No results yet. Complete a training run first.</p>';
    return;
  }
  var data = await resp.json();

  var metricsHtml = Object.entries(data.metrics)
    .map(function (e) {
      return '<div class="d-flex justify-between align-center mb-8"><span class="text-muted">' + escHtml(e[0]) + '</span><strong>' + e[1].toFixed(4) + '</strong></div>';
    }).join('');

  var maxFi = data.feature_importance ? Math.max.apply(null, Object.values(data.feature_importance)) : 1;
  var fiHtml = data.feature_importance
    ? Object.entries(data.feature_importance)
        .sort(function (a, b) { return b[1] - a[1]; })
        .map(function (e) {
          return '<div class="feat-row"><span class="feat-name">' + escHtml(e[0]) + '</span><div class="feat-bar-wrap"><div class="feat-bar" style="width:' + Math.round(e[1] / maxFi * 100) + '%"></div></div><span class="feat-score">' + e[1].toFixed(3) + '</span></div>';
        }).join('')
    : '';

  var insightsHtml = data.insights.length
    ? data.insights.map(function (i) { return '<div class="alert alert-info mb-8">' + escHtml(i) + '</div>'; }).join('')
    : '<p class="text-muted small">No issues detected.</p>';

  el.innerHTML =
    '<div class="label-sm">Metrics</div>' + metricsHtml +
    (fiHtml ? '<div class="label-sm mt-16">Feature Importance (SHAP)</div>' + fiHtml : '') +
    '<div class="label-sm mt-16">Insights</div>' + insightsHtml;
  _explainLoaded = true;
}
