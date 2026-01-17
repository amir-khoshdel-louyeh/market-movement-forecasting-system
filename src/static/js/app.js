(function(){
  // Theme management
  function initTheme(){
    const theme = localStorage.getItem('theme') || 'light';
    if(theme === 'dark') document.documentElement.classList.add('dark-mode');
    updateThemeButton();
  }
  
  function toggleTheme(){
    const isDark = document.documentElement.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeButton();
  }
  
  function updateThemeButton(){
    const btn = document.getElementById('theme-toggle');
    const isDark = document.documentElement.classList.contains('dark-mode');
    if(btn) btn.textContent = isDark ? '☀️' : '🌙';
  }
  
  const chartEl = document.getElementById('chart');
  const statusEl = document.getElementById('status');
  const symbolEl = document.getElementById('symbol');
  const intervalEl = document.getElementById('interval');
   const btnZoomIn = document.getElementById('btn-zoom-in');
   const btnZoomOut = document.getElementById('btn-zoom-out');
   const btnZoomBox = document.getElementById('btn-zoom-box');
   const btnPan = document.getElementById('btn-pan');
   const btnSelect = document.getElementById('btn-select');
   const btnLasso = document.getElementById('btn-lasso');
   const btnReset = document.getElementById('btn-reset');
   const themeToggle = document.getElementById('theme-toggle');

  let currentSymbol = (symbolEl.value || '').trim().toLowerCase();
  let currentInterval = intervalEl.value;

  let x = [], o = [], h = [], l = [], c = [], v = [];
  let es = null;

  const GREEN = 'rgba(22,163,74,0.6)';
  const RED   = 'rgba(239,68,68,0.6)';

  function setStatus(msg){
    statusEl.textContent = msg;
  }

  function volumeColors(){
    return x.map((_, i) => (c[i] >= o[i] ? GREEN : RED));
  }

  function layout(){
    return {
      template: 'plotly_white',
      hovermode: 'x unified',
      dragmode: 'zoom',
      showlegend: false,
      margin: { l: 60, r: 20, t: 40, b: 30 },
      xaxis: { domain: [0, 1], type: 'date', rangeslider: { visible: false }, showspikes: true },
      yaxis: { domain: [0.35, 1], fixedrange: false },
      xaxis2: { domain: [0, 1], anchor: 'y2', matches: 'x', showline: true, zeroline: false },
      yaxis2: { domain: [0, 0.30], fixedrange: false },
      uirevision: 'mmfs-candles',
    };
  }

  function showSkeleton(){
    const skeleton = document.getElementById('skeleton-container');
    if(skeleton) skeleton.style.display = 'flex';
  }

  function hideSkeleton(){
    const skeleton = document.getElementById('skeleton-container');
    if(skeleton) skeleton.style.display = 'none';
  }

  function render(){
    hideSkeleton();
    const candles = {
      type: 'candlestick', x, open:o, high:h, low:l, close:c,
      increasing: { line: { color: '#16a34a' }, fillcolor: 'rgba(22,163,74,0.35)' },
      decreasing: { line: { color: '#ef4444' }, fillcolor: 'rgba(239,68,68,0.35)' },
      whiskerwidth: 0.6,
      xaxis: 'x', yaxis: 'y', name: 'Candles'
    };
    const volume = {
      type: 'bar', x, y: v, marker: { color: volumeColors() }, opacity: 0.7,
      xaxis: 'x2', yaxis: 'y2', name: 'Volume'
    };
    const plotLayout = layout();
    plotLayout.autosize = true;
    Plotly.newPlot(chartEl, [candles, volume], plotLayout, { responsive: true, displayModeBar: false });
     applyDragMode('zoom');
  }

  function updateLast(){
    // Update full arrays to keep subplot axes consistent and colors in sync
    Plotly.restyle(chartEl, { open: [o], high: [h], low: [l], close: [c] }, [0]);
    Plotly.restyle(chartEl, { y: [v], 'marker.color': [volumeColors()] }, [1]);
  }

  function append(dt, open, high, low, close, vol){
    x.push(dt); o.push(open); h.push(high); l.push(low); c.push(close); v.push(vol);
    Plotly.extendTraces(chartEl, { x: [[dt]], open: [[open]], high: [[high]], low: [[low]], close: [[close]] }, [0]);
    const volColor = close >= open ? GREEN : RED;
    Plotly.extendTraces(chartEl, { x: [[dt]], y: [[vol]], 'marker.color': [[volColor]] }, [1]);
  }

  async function loadInitial(){
    showSkeleton();
    const res = await fetch('/api/candles');
    const js = await res.json();
    currentSymbol = (js.symbol || currentSymbol || '').toLowerCase();
    currentInterval = js.interval || currentInterval;
    if(currentSymbol) symbolEl.value = currentSymbol;
    if(currentInterval) intervalEl.value = currentInterval;
    const arr = js.candles || [];
    x = []; o = []; h = []; l = []; c = []; v = [];
    for(const r of arr){
      const dt = new Date(r.t);
      x.push(dt); o.push(r.o); h.push(r.h); l.push(r.l); c.push(r.c); v.push(r.v);
    }
    setStatus(`Loaded ${currentSymbol} ${currentInterval}`);
    render();
  }

  function startStream(){
    if(es){ es.close(); es = null; }
    setStatus('Connecting to stream…');
    es = new EventSource('/events');
    es.onopen = function(){ setStatus(`Streaming ${currentSymbol} ${currentInterval}`); };
    es.onmessage = function(ev){
      try{
        const m = JSON.parse(ev.data);
        if(m.type === 'hello' || m.type === 'keepalive') return;
        const dt = new Date(m.t);
        const lastIdx = x.length - 1;
        const lastX = x[lastIdx];
        if(lastX && lastX.getTime() === dt.getTime()){
          // Update in-progress candle
          h[lastIdx] = Math.max(h[lastIdx], m.h);
          l[lastIdx] = Math.min(l[lastIdx], m.l);
          c[lastIdx] = m.c;
          v[lastIdx] = m.v;
          updateLast();
        } else {
          append(dt, m.o, m.h, m.l, m.c, m.v);
        }
        setStatus(m.closed ? 'Closed candle' : 'Live…');
      }catch(e){ console.warn('SSE parse/update error', e); }
    };
    es.onerror = function(){ setStatus('Stream error'); };
  }

  async function applySelection(){
    try{
      const symbol = (symbolEl.value || '').trim().toLowerCase();
      const interval = intervalEl.value;
      currentSymbol = symbol;
      currentInterval = interval;
      setStatus('Updating selection…');
      const res = await fetch('/start', { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify({ symbol, interval }) });
      const js = await res.json();
      currentSymbol = (js.symbol || symbol).toLowerCase();
      currentInterval = js.interval || interval;
      if(currentSymbol) symbolEl.value = currentSymbol;
      if(currentInterval) intervalEl.value = currentInterval;
      await loadInitial();
      startStream();
    }catch(e){
      console.warn('Apply selection failed', e);
      setStatus('Update failed');
    }
  }

  symbolEl.addEventListener('change', applySelection);
  intervalEl.addEventListener('change', applySelection);
  if(themeToggle) themeToggle.addEventListener('click', toggleTheme);

   function normalizeRangeValue(v){
     if(v instanceof Date){ return v.getTime(); }
     if(typeof v === 'string'){ const d = new Date(v); return d.getTime(); }
     return v;
   }

   function currentXRange(){
     const layout = chartEl._fullLayout;
     if(layout && layout.xaxis && layout.xaxis.range){
       const r = layout.xaxis.range.map(normalizeRangeValue);
       if(!Number.isNaN(r[0]) && !Number.isNaN(r[1])) return r;
     }
     if(x.length >= 2){
       return [x[0].getTime(), x[x.length - 1].getTime()];
     }
     return null;
   }

   function scaleRange(range, factor){
     const [a, b] = range;
     const center = (a + b) / 2;
     const half = (b - a) / 2 * factor;
     return [center - half, center + half];
   }

   function relayoutX(range){
     Plotly.relayout(chartEl, { 'xaxis.range': range, 'xaxis.autorange': false });
   }

   function zoomIn(){
     const r = currentXRange();
     if(!r) return;
     relayoutX(scaleRange(r, 0.8));
   }

   function zoomOut(){
     const r = currentXRange();
     if(!r) return;
     relayoutX(scaleRange(r, 1 / 0.8));
   }

   function resetAxes(){
     Plotly.relayout(chartEl, { 'xaxis.autorange': true, 'yaxis.autorange': true });
   }

   const modeButtons = {
     zoom: btnZoomBox,
     pan: btnPan,
     select: btnSelect,
     lasso: btnLasso,
   };

   function setActiveButton(mode){
     Object.entries(modeButtons).forEach(([m, btn]) => {
       if(!btn) return;
       btn.classList.toggle('active', m === mode);
     });
   }

   function applyDragMode(mode){
     Plotly.relayout(chartEl, { dragmode: mode });
     setActiveButton(mode);
   }

   btnZoomIn?.addEventListener('click', zoomIn);
   btnZoomOut?.addEventListener('click', zoomOut);
   btnReset?.addEventListener('click', resetAxes);
   btnZoomBox?.addEventListener('click', () => applyDragMode('zoom'));
   btnPan?.addEventListener('click', () => applyDragMode('pan'));
   btnSelect?.addEventListener('click', () => applyDragMode('select'));
   btnLasso?.addEventListener('click', () => applyDragMode('lasso'));

  // Initialize theme and start app
  initTheme();
  loadInitial().then(startStream);
})();
