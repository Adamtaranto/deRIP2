
(function () {
  var panels = Array.prototype.slice.call(
    document.querySelectorAll('.seq-panel'));
  if (!panels.length) return;
  var indicator = document.getElementById('seq-indicator');
  var current = 0;
  var savedLeft = 0;      // shared horizontal offset for the column figures
  var syncing = false;    // guard against scroll-event feedback while syncing

  function colScrollers(panel) {
    return Array.prototype.slice.call(panel.querySelectorAll('.col-scroll'));
  }

  // Page 0 is the alignment overview; the rest are sequences 1..N.
  var nSeqs = panels.length - 1;
  function show(k) {
    current = (k + panels.length) % panels.length;
    panels.forEach(function (p, i) {
      if (i === current) { p.removeAttribute('hidden'); }
      else { p.setAttribute('hidden', ''); }
    });
    if (indicator) {
      indicator.textContent = current === 0
        ? 'Overview'
        : 'Sequence ' + current + ' / ' + nSeqs;
    }
    // Re-apply the remembered horizontal offset; leave the vertical scroll be.
    syncing = true;
    colScrollers(panels[current]).forEach(function (el) { el.scrollLeft = savedLeft; });
    syncing = false;
  }

  // Remember the horizontal offset whenever the user scrolls a column figure,
  // and mirror it to the other column figures in the same panel.
  panels.forEach(function (panel) {
    colScrollers(panel).forEach(function (el) {
      el.addEventListener('scroll', function () {
        if (syncing) return;
        savedLeft = el.scrollLeft;
        syncing = true;
        colScrollers(panel).forEach(function (other) {
          if (other !== el) { other.scrollLeft = savedLeft; }
        });
        syncing = false;
      });
    });
  });

  document.getElementById('seq-prev').addEventListener('click', function () {
    show(current - 1);
  });
  document.getElementById('seq-next').addEventListener('click', function () {
    show(current + 1);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { show(current - 1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { show(current + 1); e.preventDefault(); }
  });

  // Simultaneous zoom for every column-aligned figure (alignment row + strand
  // bias) and the overview alignment image, across all panels, so they scale
  // together. Each panel carries its own zoom control (in its sticky header);
  // the controls are class-based and kept in sync. Base pixel width comes from
  // an SVG's point size (1pt = 4/3 px) or an image's natural width.
  var zoom = 1;
  // Both the column strips and the overview are inline SVG; scale them all by
  // their intrinsic point width (1pt = 4/3 px).
  var zoomSvgs = Array.prototype.slice.call(
    document.querySelectorAll('.col-scroll svg, .aln-scroll svg'));
  function svgBasePx(svg) {
    var w = parseFloat(svg.getAttribute('width') || '0');
    return w * 4 / 3;  // pt -> css px
  }
  function applyZoom() {
    zoomSvgs.forEach(function (svg) {
      svg.style.width = (svgBasePx(svg) * zoom) + 'px';
    });
    document.querySelectorAll('.zlabel').forEach(function (l) {
      l.textContent = Math.round(zoom * 100) + '%';
    });
  }
  document.querySelectorAll('.zoom-in').forEach(function (b) {
    b.addEventListener('click', function () {
      zoom = Math.min(zoom * 1.25, 8); applyZoom();
    });
  });
  document.querySelectorAll('.zoom-out').forEach(function (b) {
    b.addEventListener('click', function () {
      zoom = Math.max(zoom / 1.25, 0.25); applyZoom();
    });
  });
  applyZoom();

  // Custom annotation tooltip: show the hovered group's data-tip with no delay,
  // follow the cursor, and pin it on click (the only path on touch devices).
  var tip = document.getElementById('psr-tip');
  var pinned = false;
  function placeTip(e) {
    var pad = 12;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth) { x = e.clientX - w - pad; }
    if (y + h > window.innerHeight) { y = e.clientY - h - pad; }
    tip.style.left = Math.max(0, x) + 'px';
    tip.style.top = Math.max(0, y) + 'px';
  }
  function showTip(text, e) {
    tip.textContent = text;
    tip.removeAttribute('hidden');
    placeTip(e);
  }
  function hideTip() {
    if (pinned) return;
    tip.setAttribute('hidden', '');
  }
  if (tip) {
    document.addEventListener('mouseover', function (e) {
      if (pinned) return;
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) { showTip(g.getAttribute('data-tip'), e); }
    });
    document.addEventListener('mousemove', function (e) {
      if (pinned || tip.hasAttribute('hidden')) return;
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) { placeTip(e); } else { tip.setAttribute('hidden', ''); }
    });
    document.addEventListener('mouseout', function (e) {
      if (pinned) return;
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) { hideTip(); }
    });
    document.addEventListener('click', function (e) {
      // A group with a FASTA payload opens the popup instead of pinning a tip.
      var fa = e.target.closest && e.target.closest('[data-fasta]');
      if (fa) {
        openFastaModal(fa.getAttribute('data-fasta'));
        pinned = false; tip.setAttribute('hidden', '');
        return;
      }
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) {
        pinned = true; showTip(g.getAttribute('data-tip'), e);
      } else {
        pinned = false; tip.setAttribute('hidden', '');
      }
    });
  }

  // Click-to-view FASTA popup. The payloads are embedded as JSON; each clickable
  // group (a CDS annotation, or the deRIP consensus row) carries a data-fasta key.
  var fastaData = {};
  var dataEl = document.getElementById('psr-fasta-data');
  if (dataEl) { try { fastaData = JSON.parse(dataEl.textContent); } catch (err) {} }
  var modal = document.getElementById('psr-modal');
  var modalTitle = document.getElementById('psr-modal-title');
  var fastaPre = document.getElementById('psr-fasta');
  var tabStrip = document.getElementById('psr-tabs');
  var modalNote = document.getElementById('psr-note');
  var copyBtn = document.getElementById('psr-copy');
  var fastaCurrent = null;   // the active payload
  var activeTab = 0;         // index into fastaCurrent.tabs

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (err) {}
    document.body.removeChild(ta);
  }

  // Rebuild the tab strip for the payload being shown. A single-tab payload (a
  // bare nucleotide record) hides the strip rather than showing one lone button.
  function buildTabs() {
    tabStrip.textContent = '';
    var tabs = fastaCurrent.tabs;
    if (tabs.length < 2) { tabStrip.setAttribute('hidden', ''); return; }
    tabStrip.removeAttribute('hidden');
    tabs.forEach(function (tab, i) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'psr-tab' + (i === activeTab ? ' is-active' : '');
      btn.textContent = tab.label;
      btn.addEventListener('click', function () {
        activeTab = i; buildTabs(); renderFastaTab();
      });
      tabStrip.appendChild(btn);
    });
  }

  function renderFastaTab() {
    if (!fastaCurrent) return;
    var tab = fastaCurrent.tabs[activeTab];
    // A tab carrying pre-rendered HTML draws its highlighted form; the plain
    // text is the same characters, so the element's textContent (what Copy
    // reads) is the unformatted record either way.
    if (tab.html) { fastaPre.innerHTML = tab.html; }
    else { fastaPre.textContent = tab.text; }
    if (tab.note) {
      modalNote.textContent = tab.note;
      modalNote.removeAttribute('hidden');
    } else {
      modalNote.setAttribute('hidden', '');
    }
  }

  function openFastaModal(key) {
    if (!modal || !fastaData[key]) return;
    fastaCurrent = fastaData[key];
    activeTab = 0;
    modalTitle.textContent = fastaCurrent.name;
    buildTabs();
    renderFastaTab();
    modal.removeAttribute('hidden');
  }
  function closeFastaModal() {
    if (modal) { modal.setAttribute('hidden', ''); }
    fastaCurrent = null;
  }

  if (modal) {
    modal.addEventListener('click', function (e) {
      if (e.target.closest('[data-close]')) { closeFastaModal(); }
    });
    copyBtn.addEventListener('click', function () {
      var text = fastaPre.textContent;
      function done() {
        var old = copyBtn.textContent; copyBtn.textContent = 'Copied!';
        setTimeout(function () { copyBtn.textContent = old; }, 1200);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {
          fallbackCopy(text); done();
        });
      } else { fallbackCopy(text); done(); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !modal.hasAttribute('hidden')) { closeFastaModal(); }
    });
  }

  // Sortable overview stats table. Each sortable header carries its column index
  // (data-ci); a click sorts the tbody rows by that column. Cells are compared
  // numerically when both parse as numbers (a leading '+' is stripped), otherwise
  // as text; en-dash / empty cells (not-applicable stats) always sort last.
  function cellVal(td) {
    var t = (td.textContent || '').trim();
    if (t === '' || t === '\u2013' || t === '\u2014') { return { n: null, s: '' }; }
    var n = parseFloat(t.replace('+', ''));
    return isNaN(n) ? { n: null, s: t } : { n: n, s: t };
  }
  var statsTable = document.querySelector('table.psr-stats');
  if (statsTable && statsTable.tBodies.length) {
    var statsBody = statsTable.tBodies[0];
    var sortDir = null, sortCi = null;
    Array.prototype.slice.call(
      statsTable.querySelectorAll('th.sortable')).forEach(function (th) {
      th.addEventListener('click', function () {
        var ci = parseInt(th.getAttribute('data-ci'), 10);
        var asc = !(sortCi === ci && sortDir === 'asc');
        sortCi = ci; sortDir = asc ? 'asc' : 'desc';
        var rows = Array.prototype.slice.call(statsBody.rows);
        rows.sort(function (a, b) {
          var x = cellVal(a.cells[ci]), y = cellVal(b.cells[ci]);
          if (x.n === null && y.n === null) {
            return asc ? x.s.localeCompare(y.s) : y.s.localeCompare(x.s);
          }
          if (x.n === null) { return 1; }   // not-applicable sorts last
          if (y.n === null) { return -1; }
          return asc ? x.n - y.n : y.n - x.n;
        });
        rows.forEach(function (r) { statsBody.appendChild(r); });
        Array.prototype.slice.call(
          statsTable.querySelectorAll('th.sortable')).forEach(function (h) {
          h.removeAttribute('data-dir');
        });
        th.setAttribute('data-dir', asc ? 'asc' : 'desc');
      });
    });
  }

  // Sortable per-motif flank-context data tables (one per sequence panel plus
  // the overview). The motif column sorts by its 5' (first) or 3' (last) flanking
  // base; numeric columns sort by their data-val, with blank (%-not-applicable)
  // cells always last. Delegated so every .flank-data table is handled.
  function flankNumVal(td) {
    var v = td.getAttribute('data-val');
    if (v === null || v === '') { return null; }
    var n = parseFloat(v);
    return isNaN(n) ? null : n;
  }
  function flankReorder(table, keyFn, numeric, asc) {
    var body = table.tBodies[0];
    if (!body) { return; }
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = keyFn(a), y = keyFn(b);
      if (numeric) {
        if (x === null && y === null) { return 0; }
        if (x === null) { return 1; }   // not-applicable sorts last
        if (y === null) { return -1; }
        return asc ? x - y : y - x;
      }
      return asc ? x.localeCompare(y) : y.localeCompare(x);
    });
    rows.forEach(function (r) { body.appendChild(r); });
  }
  function flankToggle(table, key) {
    var asc = !(table.getAttribute('data-sortkey') === key
                && table.getAttribute('data-sortdir') === 'asc');
    table.setAttribute('data-sortkey', key);
    table.setAttribute('data-sortdir', asc ? 'asc' : 'desc');
    return asc;
  }
  function flankClearHeaders(table) {
    Array.prototype.slice.call(table.querySelectorAll('th')).forEach(function (h) {
      h.removeAttribute('data-dir');
    });
    Array.prototype.slice.call(
      table.querySelectorAll('.motif-sort button')).forEach(function (b) {
      b.removeAttribute('data-active');
    });
  }
  document.addEventListener('click', function (e) {
    var mb = e.target.closest && e.target.closest('.flank-data [data-motifsort]');
    if (mb) {
      var table = mb.closest('table.flank-data');
      var which = mb.getAttribute('data-motifsort');       // 'first' | 'last'
      var attr = which === 'first' ? 'data-first' : 'data-last';
      var asc = flankToggle(table, 'motif-' + which);
      flankReorder(table, function (row) {
        var td = row.cells[0];
        return (td.getAttribute(attr) || '') + td.textContent;
      }, false, asc);
      flankClearHeaders(table);
      mb.setAttribute('data-active', '1');
      return;
    }
    var th = e.target.closest && e.target.closest('table.flank-data th.sortable-num');
    if (th) {
      var t2 = th.closest('table.flank-data');
      var ci = Array.prototype.indexOf.call(th.parentNode.cells, th);
      var asc2 = flankToggle(t2, 'col-' + ci);
      flankReorder(t2, function (row) {
        return flankNumVal(row.cells[ci]);
      }, true, asc2);
      flankClearHeaders(t2);
      th.setAttribute('data-dir', asc2 ? 'asc' : 'desc');
    }
  });

  // Sequence names in the overview stats table jump to that sequence's page.
  // Scroll position is preserved (as with arrow-key navigation); the sticky nav
  // and panel header keep the reader oriented.
  document.addEventListener('click', function (e) {
    var link = e.target.closest && e.target.closest('[data-goto]');
    if (!link) { return; }
    e.preventDefault();
    show(parseInt(link.getAttribute('data-goto'), 10));
  });

  // Land on the overview with the whole MSA figure visible: shrink the shared
  // zoom just enough to fit the alignment figure in its scroll box (never zoom in
  // past 100%). Widths are then applied by applyZoom below.
  function fitOverview() {
    var box = document.querySelector(
      '.seq-panel[data-index="overview"] .aln-scroll');
    if (!box) { return; }
    var svg = box.querySelector('svg');
    if (!svg) { return; }
    var base = svgBasePx(svg);
    var avail = box.clientWidth - 12;   // minus the box padding
    if (base > 0 && avail > 0 && avail < base) {
      zoom = Math.max(0.25, avail / base);
    }
  }

  show(0);
  fitOverview();
  applyZoom();
})();
