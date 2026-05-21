(function () {
  var saved = localStorage.getItem('mb-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);

  function toggleTheme() {
    var current = document.documentElement.getAttribute('data-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('mb-theme', next);
    var btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = next === 'dark' ? '☀ Light' : '☾ Dark';
  }

  window.toggleTheme = toggleTheme;

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle');
    var current = document.documentElement.getAttribute('data-theme');
    if (btn) btn.textContent = current === 'dark' ? '☀ Light' : '☾ Dark';
  });
})();
