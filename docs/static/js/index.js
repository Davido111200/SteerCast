document.addEventListener('DOMContentLoaded', function () {
  document.documentElement.classList.add('js');

  // Equations
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  }

  // Results tabs
  var tabs = Array.prototype.slice.call(document.querySelectorAll('[data-tab]'));
  function activate(target) {
    tabs.forEach(function (tab) {
      var on = tab.getAttribute('data-tab') === target;
      tab.parentElement.classList.toggle('is-active', on);
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      var panel = document.getElementById(tab.getAttribute('data-tab'));
      if (panel) panel.classList.toggle('is-hidden', !on);
    });
  }
  tabs.forEach(function (tab) {
    tab.addEventListener('click', function (event) {
      event.preventDefault();
      activate(tab.getAttribute('data-tab'));
    });
  });
  if (tabs.length) {
    // Deep links such as index.html#table-fixed open the matching tab.
    var initial = tabs[0].getAttribute('data-tab');
    tabs.forEach(function (tab) {
      if (window.location.hash && tab.getAttribute('href') === window.location.hash) {
        initial = tab.getAttribute('data-tab');
      }
    });
    activate(initial);
  }

  // Copy BibTeX
  var button = document.getElementById('copy-bibtex');
  var source = document.getElementById('bibtex-text');
  if (button && source) {
    button.addEventListener('click', function () {
      var text = source.innerText;
      var label = button.querySelector('span:last-child');
      function done(ok) {
        label.textContent = ok ? 'Copied' : 'Copy failed';
        setTimeout(function () { label.textContent = 'Copy'; }, 1800);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
      } else {
        var area = document.createElement('textarea');
        area.value = text;
        document.body.appendChild(area);
        area.select();
        var ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(area);
        done(ok);
      }
    });
  }
});
