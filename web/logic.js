// Replaces the mockup's constants. Everything clinical now comes from /api/*; only the nav
// labels and the empty-state copy are held here, because those are interface chrome rather
// than patient data.
const SCREENS = [
  { id: 'home', label: 'Home', icon: '⌂' },
  { id: 'intake', label: 'New assessment', icon: '＋' },
  { id: 'clinical', label: 'Clinical data', icon: '❤' },
  { id: 'pocus', label: 'POCUS', icon: '◉' },
  { id: 'diagnosis', label: 'Diagnosis', icon: '✳' },
  { id: 'record', label: 'Patient record', icon: '▣' },
  { id: 'assessment', label: 'Assessment', icon: '✦', badge: 'AI' },
  { id: 'alerts', label: 'Alerts', icon: '⚠' },
  { id: 'assistant', label: 'Clinical assistant', icon: '✧' },
  { id: 'timeline', label: 'Timeline', icon: '◷' },
  { id: 'report', label: 'Report', icon: '▤' },
  { id: 'history', label: 'History', icon: '⟲' }
];

// Screens that read a computed encounter. Without one they have nothing to show, so the nav
// marks them rather than letting a click land on an empty page and read as a broken button.
const NEEDS_ENCOUNTER = ['diagnosis', 'assessment', 'alerts', 'assistant', 'timeline', 'report'];

const SUGGESTIONS = [
  'What findings support the leading entry?',
  'What information is missing?',
  'What did POCUS actually see?',
  'Summarise this patient.',
  'What should I investigate next?',
  'Challenge this assessment.'
];

const post = (url, body) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify(body) }).then(r => r.json());

class Component extends DCLogic {
  state = {
    screen: 'home', recTab: 'images', draft: '', busy: false, grown: false,
    boot: null, view: null, preset: '', upload: null,
    form: { name: '', age: 60, sex: 'F', complaint: '', tier: 'medium', tconf: 0.8,
            organ: 'Lung', vitals: {}, labs: {}, findings: {} },
    messages: [{ role: 'a', text: 'Ask me anything about this encounter. I answer from the '
                                 + 'computed assessment — the state, the triggers, the alerts, '
                                 + 'the evidence and the differential as validated.' }]
  };

  componentDidMount() {
    this.load();
    requestAnimationFrame(() => requestAnimationFrame(() => this.setState({ grown: true })));
  }

  async load() {
    const [boot, view] = await Promise.all([
      fetch('/api/bootstrap').then(r => r.json()),
      fetch('/api/view').then(r => r.json())
    ]);
    this.setState({ boot, view });
  }

  go(id) { return () => this.setState({ screen: id }); }

  setForm(patch) { this.setState(s => ({ form: Object.assign({}, s.form, patch) })); }

  // ---- the pipeline ---------------------------------------------------------------
  async analyse() {
    this.setState({ busy: true });
    const f = this.state.form;
    const body = Object.assign({}, f, {
      preset: this.state.preset || null,
      reportJson: this.state.upload ? this.state.upload.report : null
    });
    const view = await post('/api/analyse', body);
    const boot = await fetch('/api/bootstrap').then(r => r.json());
    this.setState({ view, boot, busy: false, screen: 'assessment',
                    messages: this.state.messages.slice(0, 1) });
  }

  async loadPreset(key) {
    // Analysed server-side as the CANONICAL record. Posting a partial form instead produced
    // an encounter with one citable fact and thirteen absent values -- a different record from
    // the scenario it was named after, whose recorded differential was then correctly refused.
    this.setState({ busy: true });
    const view = await fetch('/api/preset?key=' + encodeURIComponent(key))
      .then(r => r.json());
    const boot = await fetch('/api/bootstrap').then(r => r.json());
    this.setState({ preset: key, view, boot, busy: false, screen: 'assessment',
                    messages: this.state.messages.slice(0, 1) });
  }

  async upload(files, which) {
    if (!files || !files.length) return;
    const b64 = await new Promise(res => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.readAsDataURL(files[0]);
    });
    if (which === 'es') { this.setState({ es: b64 }); return; }
    this.setState({ busy: true });
    const out = await post('/api/upload', { organ: this.state.form.organ, image: b64,
                                            image2: this.state.es || null });
    this.setState({ upload: out, busy: false, preview: b64 });
  }

  async ask(q) {
    const text = (q || '').trim();
    if (!text) return;
    this.setState(s => ({ screen: 'assistant', draft: '', busy: true,
                          messages: s.messages.concat([{ role: 'd', text }]) }));
    const out = await post('/api/ask', { question: text });
    this.setState(s => ({ busy: false,
                          messages: s.messages.concat([{ role: 'a', text: out.answer }]) }));
  }

  navStyle(id, disabled) {
    const on = this.state.screen === id;
    return {
      appearance: 'none', border: 0, cursor: disabled ? 'not-allowed' : 'pointer',
      textAlign: 'left', width: '100%', display: 'flex', alignItems: 'center',
      justifyContent: 'space-between', gap: '10px', padding: '12px 16px', borderRadius: '8px',
      background: on ? '#E9E8FB' : 'transparent',
      color: on ? '#2E2A78' : (disabled ? '#B9B5EC' : '#6A6785'),
      fontWeight: on ? 700 : 500, fontSize: '14.5px',
      boxShadow: on ? 'inset 3px 0 0 #5B54D6' : 'none',
      opacity: disabled ? 0.55 : 1
    };
  }

  tone(sev) {
    return sev === 'HIGH' ? { bg: '#FDECEC', fg: '#C13238' }
         : sev === 'MODERATE' ? { bg: '#FFF3E0', fg: '#9A6207' }
         : { bg: '#F0FADB', fg: '#5A7A0F' };
  }

  renderVals() {
    const st = this.state;
    const v = st.view || {};
    const boot = st.boot || {};
    const has = !!v.hasEncounter;
    const sev = v.severity || '—';
    const tone = this.tone(sev);
    const nAlerts = (v.alerts || []).length;

    const out = {
      ready: !!st.boot,
      busy: st.busy,
      hasEncounter: has,
      noEncounter: !has,
      emptyMessage: v.message || 'No encounter has been analysed yet.',

      nav: SCREENS.map(s => {
        const disabled = NEEDS_ENCOUNTER.indexOf(s.id) >= 0 && !has;
        const badge = s.id === 'alerts' && has && nAlerts ? String(nAlerts) : (s.badge || '');
        return {
          label: s.label, icon: s.icon, badge: badge,
          style: this.navStyle(s.id, disabled),
          onClick: disabled ? (() => {}) : this.go(s.id),
          badgeStyle: badge
            ? { background: s.id === 'alerts' ? '#FDECEC' : '#E4E2F8',
                color: s.id === 'alerts' ? '#C13238' : '#5B3CC4', borderRadius: '999px',
                padding: '3px 9px', fontSize: '11.5px', fontWeight: 700 }
            : { display: 'none' }
        };
      }),

      // ---- patient -----------------------------------------------------------------
      pName: (v.patient || {}).name || 'No patient',
      pAge: (v.patient || {}).age || '—',
      pSex: (v.patient || {}).sex || '',
      pComplaint: (v.patient || {}).complaint || '—',
      pOrgan: (v.patient || {}).organ || '—',
      pId: (v.patient || {}).id || '—',
      severity: sev,
      severityStyle: { background: tone.bg, color: tone.fg, borderRadius: '999px',
                       padding: '5px 12px', fontSize: '12.5px', fontWeight: 700 },
      severityBandStyle: { background: tone.bg, border: '1px solid ' + tone.fg + '33',
                           borderLeft: '5px solid ' + tone.fg, borderRadius: '18px',
                           padding: '20px 24px' },
      severityFg: { color: tone.fg, fontSize: '12.5px', letterSpacing: '.11em',
                    textTransform: 'uppercase', fontWeight: 800 },
      conclusion: v.conclusion || '',
      scenario: v.scenario || '—',
      thresholdsVersion: v.thresholdsVersion || '—',
      escalateText: v.escalate ? 'escalation required' : 'no escalation',
      triggers: (v.triggers || []).map(t => ({ text: t })),
      caseQuality: v.caseQuality || '—',

      // ---- measurements ------------------------------------------------------------
      vitals: (v.vitals || []).map(x => ({
        label: x.label, unit: x.unit,
        value: x.value === null ? 'Not measured' : x.value,
        note: x.value === null ? 'Not the same as normal' : (x.flag || ''),
        style: x.value === null
          ? { background: '#FAFAFE', border: '1px dashed #DCD4F7', borderRadius: '18px',
              padding: '20px 22px' }
          : { background: '#fff', border: '1px solid #E4E2F8', borderRadius: '18px',
              padding: '20px 22px',
              borderTop: '4px solid ' + (x.flag === 'normal' ? '#22A06B' : '#E5484D') },
        valueStyle: { fontSize: x.value === null ? '20px' : '32px', fontWeight: 800,
                      letterSpacing: '-.02em',
                      color: x.value === null ? '#9C99B8' : '#1B1A3A' }
      })),
      labs: (v.labs || []).map(x => ({
        name: x.name, result: x.result || '—', ref: x.ref,
        status: x.result === null ? 'Not measured' : (x.flag || ''),
        rowStyle: { padding: '13px 0', color: x.result === null ? '#9C99B8' : '#1B1A3A' },
        statusStyle: {
          borderRadius: '999px', padding: '4px 11px', fontSize: '12.5px', fontWeight: 700,
          background: x.result === null ? '#EFEEFB'
                    : x.flag === 'normal' ? '#F0FADB' : '#FDECEC',
          color: x.result === null ? '#5B3CC4' : x.flag === 'normal' ? '#5A7A0F' : '#C13238'
        }
      })),
      missing: (v.missing || []).map(m => ({ name: m })),
      missingCount: (v.missing || []).length,

      // ---- imaging -----------------------------------------------------------------
      findings: (v.findings || []).map(f => ({
        label: f.label, caption: f.caption, conf: f.conf.toFixed(2),
        status: f.detected ? 'Detected' : 'Not detected',
        statusStyle: f.detected
          ? { background: '#E4E2F8', color: '#2E2A78', borderRadius: '999px',
              padding: '5px 12px', fontSize: '12.5px', fontWeight: 700 }
          : { color: '#6A6785', fontSize: '13.5px' }
      })),
      notAssessed: (v.notAssessed || []).map(o => ({ organ: o })),
      outOfScope: (v.outOfScope || []).map(s => ({ text: s })),
      topFinding: v.topFinding || '—',
      uploadRows: (st.upload ? st.upload.rows : []).map(r => ({
        label: r.label, caption: r.caption, conf: r.conf.toFixed(2),
        status: r.detected ? 'Detected' : 'Not detected'
      })),
      uploadSeconds: st.upload ? String(st.upload.seconds) : '',
      uploadModel: st.upload ? (st.upload.report.model || 'the module') : '',
      uploadScope: st.upload ? ((st.upload.report.reliability || {}).scope || '') : '',
      hasUpload: !!st.upload,
      preview: st.preview || '',

      // ---- reasoning ---------------------------------------------------------------
      differential: (v.differential || []).map((d, i) => ({
        rank: String(i + 1).padStart(2, '0'),
        diagnosis: d.diagnosis, likelihood: d.likelihood,
        supporting: (d.supporting || []).map(t => ({ text: t })),
        contradicting: (d.contradicting || []).map(t => ({ text: t })),
        limitations: (d.limitations || []).map(t => ({ text: t })),
        ids: (d.supportingIds || []).join(', ') || '—',
        style: i === 0
          ? { border: '1px solid #DEDCF4', borderRadius: '16px', padding: '20px 22px',
              background: '#FCFBFF', marginBottom: '16px' }
          : { border: '1px solid #E4E2F8', borderRadius: '16px', padding: '20px 22px',
              marginBottom: '16px' }
      })),
      hasDifferential: (v.differential || []).length > 0,
      noDifferential: (v.differential || []).length === 0,
      differentialNote: v.differentialOrigin === 'failed'
        ? 'The model backend failed and the answer was withheld. The severity and alerts beside '
          + 'it were computed before the model ran and are unaffected.'
        : v.differentialOrigin === 'not_generated'
        ? 'No differential was generated. Producing one requires a 4.9 GB language model on a '
          + 'GPU, which is not loaded in this deployment. Everything else on screen was '
          + 'computed here.'
        : '',
      withheld: !!v.withheld,
      validationErrors: (v.validationErrors || []).map(t => ({ text: t })),
      warnings: (v.warnings || []).map(t => ({ text: t })),
      evidence: (v.evidence || []).map(e => ({
        id: e.id, text: e.text, mark: e.cited ? '●' : '○',
        style: { color: e.cited ? '#1B1A3A' : '#6A6785' }
      })),
      hits: (v.hits || []).map(h => ({
        n: String(h.n), id: h.id, topic: h.topic, score: h.score.toFixed(2),
        source: h.source, text: h.text
      })),
      noHits: (v.hits || []).length === 0,

      // ---- alerts ------------------------------------------------------------------
      alerts: (v.alerts || []).map(a => {
        const crit = a.severity === 'CRITICAL';
        return {
          type: a.type, message: a.message,
          kicker: crit ? 'Immediate attention' : 'Important',
          style: { background: '#fff', borderRadius: '18px', padding: '22px 26px',
                   marginBottom: '16px',
                   border: '1px solid ' + (crit ? '#F6CFCF' : '#F7DFB4'),
                   borderLeft: '5px solid ' + (crit ? '#E5484D' : '#F5A623') },
          kickerStyle: { color: crit ? '#C13238' : '#9A6207', fontSize: '12.5px',
                         letterSpacing: '.11em', textTransform: 'uppercase',
                         fontWeight: 800 }
        };
      }),
      noAlerts: (v.alerts || []).length === 0,
      alertCount: String(nAlerts),

      // ---- recommendations ---------------------------------------------------------
      exams: (v.exams || []).map(e => ({
        exam: e.exam, reason: e.reason, priority: e.priority
      })),
      nextStep: (v.exams || []).length
        ? v.exams[0].exam : 'Physician review of the current findings.',
      therapeutic: ((v.therapeutic || {}).considerations || []).map(c => ({
        consideration: c.consideration, basis: c.basis, passage: c.passage,
        disclaimer: c.disclaimer
      })),
      noTherapeutic: !((v.therapeutic || {}).considerations || []).length,
      therapeuticStatus: (v.therapeutic || {}).status || '',

      // ---- bar chart: counts of cited evidence, not probabilities -------------------
      bars: (v.bars || []).map(b => {
        const total = Math.max(b.sup + b.ag, 1);
        const split = Math.round(b.sup / total * 100);
        const maxN = Math.max.apply(null, (v.bars || []).map(x => Math.max(x.sup, x.ag)).concat([1]));
        return {
          label: b.label,
          barStyle: {
            width: '100%', maxWidth: '46px',
            height: st.grown ? Math.round(Math.max(b.sup, b.ag) / maxN * 100) + '%' : '0%',
            transition: 'height .8s cubic-bezier(.2,.8,.25,1)',
            background: 'linear-gradient(180deg,#5B54D6 0 ' + split + '%,#2E2A78 ' + split + '% 100%)'
          }
        };
      }),
      noBars: (v.bars || []).length === 0,

      // ---- timeline ----------------------------------------------------------------
      timeline: (v.timeline || []).map(t => ({
        time: t.time, title: t.title, detail: t.detail,
        dotStyle: { width: '13px', height: '13px', borderRadius: '50%', marginTop: '5px',
                    background: t.hl ? '#5B3CC4' : '#fff',
                    border: t.hl ? '3px solid #D9CDFF' : '3px solid #DEDCF4',
                    display: 'block' },
        titleStyle: { fontWeight: t.hl ? 800 : 600, fontSize: '15.5px',
                      color: t.hl ? '#2E2A78' : '#1B1A3A' }
      })),

      // ---- report ------------------------------------------------------------------
      reportText: v.reportText || '',
      generatedAt: v.generatedAt || '',

      // ---- roster / record ---------------------------------------------------------
      roster: (boot.roster || []).map(r => ({
        name: r.name, age: String(r.age), sex: r.sex, complaint: r.complaint,
        tag: r.tag, severity: r.severity, alerts: r.alerts + ' alert(s)',
        initials: r.name.split(' ').map(w => w[0]).join('').toUpperCase(),
        onOpen: () => this.loadPreset(r.key),
        tagStyle: {
          borderRadius: '999px', padding: '5px 12px', fontSize: '12.5px', fontWeight: 700,
          background: r.tag === 'Critical' ? '#FDECEC' : r.tag === 'Review' ? '#FFF3E0' : '#F0FADB',
          color: r.tag === 'Critical' ? '#C13238' : r.tag === 'Review' ? '#9A6207' : '#5A7A0F'
        }
      })),
      records: (boot.records || []).map(r => ({
        name: r.name, at: r.at, severity: r.severity, organ: r.organ,
        encounterId: r.encounterId,
        findings: (r.findings || []).join(', ') || 'no positive finding',
        alerts: r.alerts + ' alert(s)'
      })),
      noRecords: !(boot.records || []).length,
      rosterCount: String((boot.roster || []).length),
      sessionCount: String((boot.records || []).length),
      criticalCount: String((boot.roster || []).filter(r => r.severity === 'HIGH').length),
      testCount: boot.tests ? String(boot.tests) : '—',
      modules: Object.keys(boot.modules || {}).map(k => ({
        organ: k, reason: boot.modules[k].reason,
        dot: boot.modules[k].runs ? '●' : '○'
      })),

      // ---- intake form -------------------------------------------------------------
      fName: st.form.name, fAge: String(st.form.age), fSex: st.form.sex,
      fComplaint: st.form.complaint, fTier: st.form.tier, fOrgan: st.form.organ,
      onName: e => this.setForm({ name: e.target.value }),
      onAge: e => this.setForm({ age: parseInt(e.target.value || '0', 10) }),
      onSex: e => this.setForm({ sex: e.target.value }),
      onComplaint: e => this.setForm({ complaint: e.target.value }),
      onTier: e => this.setForm({ tier: e.target.value }),
      onOrgan: e => this.setForm({ organ: e.target.value }),
      organs: (boot.organs || []).map(o => ({ name: o })),
      vitalInputs: (boot.vitalKeys || []).map(k => ({
        key: k.key, label: k.key + ' (' + k.unit + ')',
        value: st.form.vitals[k.key] === undefined ? '' : String(st.form.vitals[k.key]),
        onChange: e => {
          const t = e.target.value;
          const nv = Object.assign({}, this.state.form.vitals);
          if (t === '') delete nv[k.key]; else nv[k.key] = parseFloat(t);
          this.setForm({ vitals: nv });
        }
      })),
      labInputs: (boot.labKeys || []).map(k => ({
        key: k.key, label: k.key + (k.unit ? ' (' + k.unit + ')' : ''),
        value: st.form.labs[k.key] === undefined ? '' : String(st.form.labs[k.key]),
        onChange: e => {
          const t = e.target.value;
          const nl = Object.assign({}, this.state.form.labs);
          if (t === '') delete nl[k.key]; else nl[k.key] = parseFloat(t);
          this.setForm({ labs: nl });
        }
      })),
      findingInputs: (st.form.organ === 'Lung' ? (boot.lungFindings || [])
                     : st.form.organ === 'Heart' ? ['severe dysfunction']
                     : st.form.organ === 'Gallbladder' ? (boot.gbClasses || []) : []
                    ).map(name => ({
        name: name.replace(/_/g, ' '),
        value: st.form.findings[name] === undefined ? '' : String(st.form.findings[name]),
        onChange: e => {
          const t = e.target.value;
          const nf = Object.assign({}, this.state.form.findings);
          if (t === '') delete nf[name]; else nf[name] = parseFloat(t);
          this.setForm({ findings: nf });
        }
      })),
      onUpload: e => this.upload(e.target.files, 'ed'),
      onUploadEs: e => this.upload(e.target.files, 'es'),
      onAnalyse: () => this.analyse(),

      // ---- assistant ---------------------------------------------------------------
      messages: st.messages.map(m => ({
        text: m.text,
        rowStyle: { display: 'flex',
                    justifyContent: m.role === 'd' ? 'flex-end' : 'flex-start' },
        bubbleStyle: m.role === 'd'
          ? { maxWidth: '74%', background: '#5B54D6', color: '#fff', padding: '14px 18px',
              borderRadius: '16px 16px 4px 16px', fontSize: '14.5px', lineHeight: 1.55,
              whiteSpace: 'pre-wrap' }
          : { maxWidth: '78%', background: '#F4F3FD', border: '1px solid #E4E2F8',
              color: '#1B1A3A', padding: '14px 18px', borderRadius: '16px 16px 16px 4px',
              fontSize: '14.5px', lineHeight: 1.6, whiteSpace: 'pre-wrap' }
      })),
      suggestions: SUGGESTIONS.map(label => ({ label, onClick: () => this.ask(label) })),
      draft: st.draft,
      onDraft: e => this.setState({ draft: e.target.value }),
      onSubmit: e => { e.preventDefault(); this.ask(this.state.draft); },
      askWhy: () => this.ask('Why is this the severity?'),
      askChallenge: () => this.ask('Challenge this assessment.'),
      askBLines: () => this.ask('What did POCUS actually see?'),

      // ---- record tabs -------------------------------------------------------------
      recordTabs: [['images', 'POCUS images'], ['data', 'Measurements'],
                   ['visits', 'Encounters'], ['reports', 'Reports']].map(([id, label]) => ({
        label, onClick: () => this.setState({ recTab: id }),
        style: { appearance: 'none', cursor: 'pointer', borderRadius: '8px',
                 padding: '11px 18px', fontWeight: 700, fontSize: '14px',
                 background: st.recTab === id ? '#5B54D6' : '#fff',
                 color: st.recTab === id ? '#fff' : '#6A6785',
                 border: '1px solid ' + (st.recTab === id ? '#5B54D6' : '#DEDCF4') }
      })),
      recImages: st.recTab === 'images',
      recData: st.recTab === 'data',
      recVisits: st.recTab === 'visits',
      recReports: st.recTab === 'reports'
    };

    SCREENS.forEach(s => {
      const cap = s.id[0].toUpperCase() + s.id.slice(1);
      out['on' + cap] = st.screen === s.id;
      out['go' + cap] = this.go(s.id);
    });
    return out;
  }
}
