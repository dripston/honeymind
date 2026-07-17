import { useState, useEffect, useRef } from 'react';
import './index.css';

function TerminalWindow({ output, large }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [output]);

  return (
    <div className={`terminal-wrapper ${large ? 'large-term' : 'small-term'}`}>
      <div className="terminal-content" ref={containerRef}>
        {output.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
        <div className="terminal-cursor"></div>
      </div>
    </div>
  );
}

function App() {
  const [stats, setStats] = useState({ total_sessions: 0, attack_sessions: 0, poisoned_responses: 0, attack_breakdown: {} });
  const [defenseEnabled, setDefenseEnabled] = useState(true);

  // Victim Terminal States
  const [trainOutput, setTrainOutput] = useState(["> Awaiting training command..."]);
  const [victimLogs, setVictimLogs] = useState(["> Connecting to Victim API access logs..."]);
  const [gatewayLogs, setGatewayLogs] = useState(["> Connecting to HoneyMind Gateway logs..."]);

  // Attack Checkboxes
  const [undefAttacks, setUndefAttacks] = useState({ knockoff: true, jbda: true, analytical: true, evolutionary: true });
  const [defAttacks, setDefAttacks] = useState({ knockoff: true, jbda: true, analytical: true, evolutionary: true });

  // Attack Terminal States (Undefended)
  const [unDefKnockoff, setUnDefKnockoff] = useState(["> Awaiting..."]);
  const [unDefJbda, setUnDefJbda] = useState(["> Awaiting..."]);
  const [unDefAnalytical, setUnDefAnalytical] = useState(["> Awaiting..."]);
  const [unDefEvo, setUnDefEvo] = useState(["> Awaiting..."]);
  const [unDefEvalOutput, setUnDefEvalOutput] = useState(["> Awaiting Stolen Model Training..."]);

  // Attack Terminal States (Defended)
  const [defKnockoff, setDefKnockoff] = useState(["> Awaiting..."]);
  const [defJbda, setDefJbda] = useState(["> Awaiting..."]);
  const [defAnalytical, setDefAnalytical] = useState(["> Awaiting..."]);
  const [defEvo, setDefEvo] = useState(["> Awaiting..."]);
  const [defEvalOutput, setDefEvalOutput] = useState(["> Awaiting Stolen Model Training..."]);

  // Legitimate Client
  const [legitOutput, setLegitOutput] = useState(["> Simulator ready. Launch traffic to test gateway integrity..."]);

  // Poll gateway stats
  useEffect(() => {
    const fetchData = async () => {
      try {
        const statsRes = await fetch('http://127.0.0.1:8001/api/v1/stats');
        setStats(await statsRes.json());
      } catch (err) {}
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  // Victim Logs
  useEffect(() => {
    const ev1 = new EventSource('http://127.0.0.1:8001/api/v1/control/victim_logs');
    ev1.onmessage = (event) => setVictimLogs(prev => [...prev.slice(-49), event.data]);
    const ev2 = new EventSource('http://127.0.0.1:8001/api/v1/control/gateway_logs');
    ev2.onmessage = (event) => setGatewayLogs(prev => [...prev.slice(-49), event.data]);
    return () => { ev1.close(); ev2.close(); };
  }, []);

  const toggleDefense = async () => {
    const newState = !defenseEnabled;
    setDefenseEnabled(newState);
    try { await fetch(`http://127.0.0.1:8001/api/v1/control/toggle?enabled=${newState}`, { method: 'POST' }); }
    catch (err) {}
  };

  const baseUrls = ['http://127.0.0.1:8001', 'http://127.0.0.2:8001', 'http://127.0.0.3:8001', 'http://localhost:8001'];
  const currentBaseIndex = useRef(0);

  const streamProcess = (endpoint, setOutput) => {
    const baseUrl = baseUrls[currentBaseIndex.current % baseUrls.length];
    currentBaseIndex.current++;

    setOutput([`> Executing: ${endpoint} via ${baseUrl}...`]);
    const eventSource = new EventSource(`${baseUrl}/api/v1/control/${endpoint}`);
    eventSource.onmessage = (event) => {
      if (event.data === '[PROCESS_COMPLETE]') {
        eventSource.close();
        setOutput(prev => [...prev, '> Process Complete.']);
      } else {
        setOutput(prev => [...prev.slice(-99), event.data]);
      }
    };
    eventSource.onerror = () => {
      eventSource.close();
      setOutput(prev => [...prev, '> Error connecting to stream.']);
    };
  };

  const launchUndefended = () => {
    if (undefAttacks.knockoff) streamProcess('attack/knockoff/undefended', setUnDefKnockoff); else setUnDefKnockoff(["> [SKIPPED]"]);
    if (undefAttacks.jbda) streamProcess('attack/jbda/undefended', setUnDefJbda); else setUnDefJbda(["> [SKIPPED]"]);
    if (undefAttacks.analytical) streamProcess('attack/analytical/undefended', setUnDefAnalytical); else setUnDefAnalytical(["> [SKIPPED]"]);
    if (undefAttacks.evolutionary) streamProcess('attack/evolutionary/undefended', setUnDefEvo); else setUnDefEvo(["> [SKIPPED]"]);
  };

  const trainUndefended = () => {
    const included = Object.keys(undefAttacks).filter(k => undefAttacks[k]).join(',');
    streamProcess(`eval/undefended?include=${included}`, setUnDefEvalOutput);
  };

  const launchDefended = () => {
    if (defAttacks.knockoff) streamProcess('attack/knockoff/defended', setDefKnockoff); else setDefKnockoff(["> [SKIPPED]"]);
    if (defAttacks.jbda) streamProcess('attack/jbda/defended', setDefJbda); else setDefJbda(["> [SKIPPED]"]);
    if (defAttacks.analytical) streamProcess('attack/analytical/defended', setDefAnalytical); else setDefAnalytical(["> [SKIPPED]"]);
    if (defAttacks.evolutionary) streamProcess('attack/evolutionary/defended', setDefEvo); else setDefEvo(["> [SKIPPED]"]);
  };

  const trainDefended = () => {
    const included = Object.keys(defAttacks).filter(k => defAttacks[k]).join(',');
    streamProcess(`eval/defended?include=${included}`, setDefEvalOutput);
  };

  const toggleAttack = (setGroup, attack) => setGroup(prev => ({ ...prev, [attack]: !prev[attack] }));
  
  const renderChecks = (group, setGroup) => (
    <div className="attack-checks">
      {['knockoff', 'jbda', 'analytical', 'evolutionary'].map(a => (
        <label key={a} className="neo-checkbox-label">
          <input type="checkbox" checked={group[a]} onChange={() => toggleAttack(setGroup, a)} />
          <span>{a.toUpperCase()}</span>
        </label>
      ))}
    </div>
  );

  return (
    <div className="app-container">
      <header className="header-bar">
        <div className="header-title">HONEYMIND // ACTIVE DEFENSE CONTROL CENTER</div>
        <div className="neo-button" style={{ background: stats.attack_sessions > 0 ? 'var(--neo-red)' : 'var(--neo-yellow)', color: stats.attack_sessions > 0 ? 'white' : 'black', cursor: 'default' }}>
          {stats.attack_sessions > 0 ? 'THREAT DETECTED' : 'SYSTEM SECURE'}
        </div>
      </header>

      <div className="dashboard-grid">
        
        {/* LEFT COLUMN */}
        <div className="column">
          {/* Victim Server Window */}
          <div className="os-window-static">
            <div className="window-titlebar">
              <span className="titlebar-text">TARGET: VICTIM SERVER</span>
              <div className="window-controls"><div className="win-btn win-min"></div><div className="win-btn win-max"></div></div>
            </div>
            <div className="window-content">
              <button className="neo-button w-100 mb-1 btn-white" onClick={() => streamProcess('train', setTrainOutput)}>Train Victim Model</button>
              <h3 className="neo-label">Live Training Terminal</h3>
              <TerminalWindow output={trainOutput} large={true} />

              <h3 className="neo-label mt-2">Victim API Server Logs</h3>
              <TerminalWindow output={victimLogs} large={true} />

              <h3 className="neo-label mt-2" style={{color: 'var(--neo-red)'}}>HoneyMind Gateway Live Monitor</h3>
              <TerminalWindow output={gatewayLogs} large={false} />
            </div>
          </div>

          {/* Legitimate Customer App Window */}
          <div className="os-window-static" style={{ minHeight: '300px' }}>
            <div className="window-titlebar" style={{background: 'var(--neo-white)', color: 'black'}}>
              <span className="titlebar-text">LEGITIMATE CUSTOMER APP (SIMULATOR)</span>
              <div className="window-controls"><div className="win-btn win-min"></div><div className="win-btn win-max"></div></div>
            </div>
            <div className="window-content" style={{ display: 'flex', flexDirection: 'column' }}>
              <button className="neo-button w-100 mb-1 btn-white" onClick={() => streamProcess('attack/legitimate/defended', setLegitOutput)}>Launch Traffic Simulator</button>
              <div style={{ flex: '1 1 0%', minHeight: 0 }}>
                <TerminalWindow output={legitOutput} large={true} />
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="column">
          {/* HoneyMind OFF Window */}
          <div className="os-window-static">
            <div className="window-titlebar">
              <span className="titlebar-text">HONEYMIND OFF (UNDEFENDED)</span>
              <div className="window-controls"><div className="win-btn win-min"></div><div className="win-btn win-max"></div></div>
            </div>
            <div className="window-content">
              <h3 className="neo-label">Select Attack Vectors:</h3>
              {renderChecks(undefAttacks, setUndefAttacks)}
              
              <button className="neo-button w-100 mb-1 btn-red" onClick={launchUndefended}>Launch Attack Extraction</button>
              
              <div className="terminals-grid">
                <div><h3 className="neo-label">Knockoff</h3><TerminalWindow output={unDefKnockoff} /></div>
                <div><h3 className="neo-label">JBDA</h3><TerminalWindow output={unDefJbda} /></div>
                <div><h3 className="neo-label">Analytical</h3><TerminalWindow output={unDefAnalytical} /></div>
                <div><h3 className="neo-label">Evolutionary</h3><TerminalWindow output={unDefEvo} /></div>
              </div>

              <button className="neo-button w-100 mt-2 mb-1 btn-white" onClick={trainUndefended}>Evaluate Stolen Models</button>
              <h3 className="neo-label">Evaluation Matrix</h3>
              <TerminalWindow output={unDefEvalOutput} large={true} />
            </div>
          </div>

          {/* HoneyMind ON Window */}
          <div className="os-window-static">
            <div className="window-titlebar" style={{background: 'var(--neo-yellow)', color: 'black'}}>
              <span className="titlebar-text">HONEYMIND ON (DEFENDED)</span>
              <div className="window-controls"><div className="win-btn win-min"></div><div className="win-btn win-max"></div></div>
            </div>
            <div className="window-content">
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem', marginBottom: '1.5rem' }}>
                <div className="switch-container" style={{ marginBottom: 0 }}>
                  <span className="neo-label" style={{ fontSize: '1rem', fontWeight: 900, marginBottom: 0 }}>Master Gateway</span>
                  <label className="switch">
                    <input type="checkbox" checked={defenseEnabled} onChange={toggleDefense} />
                    <span className="slider"></span>
                  </label>
                </div>

                <div className="metrics-grid">
                  <div className="neo-box p-1" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}><div className="neo-label">OOD</div><div className="neo-value sm-val">{stats.total_sessions}</div></div>
                  <div className="neo-box p-1 neo-yellow" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}><div className="neo-label">Atk</div><div className="neo-value sm-val">{stats.attack_sessions}</div></div>
                  <div className="neo-box p-1 neo-red" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}><div className="neo-label">Psn</div><div className="neo-value sm-val">{stats.poisoned_responses}</div></div>
                </div>
              </div>

              <h3 className="neo-label">Select Attack Vectors:</h3>
              {renderChecks(defAttacks, setDefAttacks)}
              
              <button className="neo-button w-100 mb-1 btn-red" onClick={launchDefended}>Launch Attack Extraction</button>
              
              <div className="terminals-grid">
                <div><h3 className="neo-label">Knockoff</h3><TerminalWindow output={defKnockoff} /></div>
                <div><h3 className="neo-label">JBDA</h3><TerminalWindow output={defJbda} /></div>
                <div><h3 className="neo-label">Analytical</h3><TerminalWindow output={defAnalytical} /></div>
                <div><h3 className="neo-label">Evolutionary</h3><TerminalWindow output={defEvo} /></div>
              </div>

              <button className="neo-button w-100 mt-2 mb-1 btn-white" onClick={trainDefended}>Evaluate Stolen Models</button>
              <h3 className="neo-label">Evaluation Matrix</h3>
              <TerminalWindow output={defEvalOutput} large={true} />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
