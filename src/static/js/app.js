(function(){
  const chartEl = document.getElementById('chart');
  const statusEl = document.getElementById('status');
  const symbolEl = document.getElementById('symbol');
  const intervalEl = document.getElementById('interval');

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

  function render(){
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
    Plotly.newPlot(chartEl, [candles, volume], layout(), { responsive: true, displayModeBar: true });
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

  // Initial load and start stream on page open
  loadInitial().then(startStream);
})();
