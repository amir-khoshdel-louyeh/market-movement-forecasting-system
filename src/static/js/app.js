(function(){
  const chartEl = document.getElementById('chart');
  const statusEl = document.getElementById('status');
  const symbolEl = document.getElementById('symbol');
  const intervalEl = document.getElementById('interval');
  const startBtn = document.getElementById('start');

  let x = [], o=[], h=[], l=[], c=[], v=[];
  let es = null;

  function render(){
    const layout = {
      dragmode: 'zoom',
      showlegend: false,
      xaxis: { rangeslider: {visible:false}, type: 'date' },
      yaxis: { autorange: true },
      grid: { rows: 2, columns: 1, pattern: 'independent' },
      margin: { l: 50, r: 20, t: 30, b: 30 },
      paper_bgcolor: 'white', plot_bgcolor: 'white'
    };
    const data = [
      {
        type: 'candlestick', x, open:o, high:h, low:l, close:c,
        xaxis: 'x', yaxis: 'y'
      },
      {
        type: 'bar', x, y:v, xaxis: 'x', yaxis: 'y2', marker: {color: 'rgba(100,100,255,0.5)'}, name:'Volume'
      }
    ];
    Plotly.newPlot(chartEl, data, layout, {responsive:true});
  }

  function updateLast(idx){
    Plotly.restyle(chartEl, {
      open: [[o[idx]]],
      high: [[h[idx]]],
      low: [[l[idx]]],
      close: [[c[idx]]],
      'y[1]': [[v[idx]]]
    }, [0]);
    Plotly.restyle(chartEl, {'y':[v]}, [1]);
  }

  function append(dt, open, high, low, close, vol){
    x.push(dt); o.push(open); h.push(high); l.push(low); c.push(close); v.push(vol);
    Plotly.extendTraces(chartEl, {
      x: [[dt]], open: [[open]], high: [[high]], low: [[low]], close: [[close]]
    }, [0]);
    Plotly.extendTraces(chartEl, { x: [[dt]], y: [[vol]] }, [1]);
  }

  async function loadInitial(){
    const res = await fetch('/api/candles');
    const js = await res.json();
    const arr = js.candles || [];
    x = []; o=[]; h=[]; l=[]; c=[]; v=[];
    for(const r of arr){
      const dt = new Date(r.t);
      x.push(dt); o.push(r.o); h.push(r.h); l.push(r.l); c.push(r.c); v.push(r.v);
    }
    render();
  }

  function startStream(){
    if(es){ es.close(); es = null; }
    es = new EventSource('/events');
    es.onmessage = function(ev){
      try{
        const m = JSON.parse(ev.data);
        if(m.type === 'hello' || m.type === 'keepalive') return;
        const dt = new Date(m.t);
        const lastIdx = x.length - 1;
        const lastX = x[lastIdx];
        if(lastX && lastX.getTime() === dt.getTime()){
          // Update in-progress candle
          o[lastIdx] = o[lastIdx]; // preserve
          h[lastIdx] = Math.max(h[lastIdx], m.h);
          l[lastIdx] = Math.min(l[lastIdx], m.l);
          c[lastIdx] = m.c;
          v[lastIdx] = m.v;
          updateLast(lastIdx);
        } else {
          append(dt, m.o, m.h, m.l, m.c, m.v);
        }
        statusEl.textContent = m.closed ? 'Closed candle' : 'Live…';
      }catch(e){ console.warn(e); }
    };
    es.onerror = function(){ statusEl.textContent = 'Stream error'; };
  }

  startBtn.addEventListener('click', async function(){
    const symbol = symbolEl.value.trim().toLowerCase();
    const interval = intervalEl.value;
    const res = await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({symbol, interval})});
    const js = await res.json();
    await loadInitial();
    startStream();
    statusEl.textContent = `Streaming ${js.symbol} ${js.interval}`;
  });

  // Initial load and start stream on page open
  loadInitial().then(startStream);
})();
