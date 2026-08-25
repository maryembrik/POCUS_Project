// Replaces the mockup's constants. Everything clinical comes from /api/*; only nav labels and
// empty-state copy live here, because those are interface chrome rather than patient data.
const SCREENS = [
  { id: 'home', label: 'Home', icon: '⌂' },
  { id: 'workup', label: 'Patient workup', icon: '✎' },
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
const NEEDS = ['diagnosis', 'assessment', 'alerts', 'assistant', 'timeline', 'report'];

const SUGGESTIONS = [
  'What findings support the leading entry?',
  'What information is missing?',
  'What did POCUS actually see?',
  'Summarise this patient.',
  'What should I investigate next?',
  'Challenge this assessment.'
];

const post = (u, b) => fetch(u, { method: 'POST',
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })
  .then(r => r.json());
const get = u => fetch(u).then(r => r.json());

class Component extends DCLogic {
  state = {
    screen: 'home', recTab: 'images', draft: '', busy: false, grown: false, fabOpen: false,
    boot: null, view: null, preset: '', upload: null, previews: [],
    form: { name: '', age: 60, sex: 'F', complaint: '', history: '', tier: 'medium',
            tconf: 0.8, organ: 'Lung', vitals: {}, labs: {}, findings: {} },
    messages: [{ role: 'a', text: 'Ask me anything about this encounter. I answer from the '
      + 'computed assessment — the clinical state, the escalation triggers, the alerts, the '
      + 'evidence and the differential as validated.' }]
  };

  componentDidMount() {
    this.load();
    requestAnimationFrame(() => requestAnimationFrame(() => this.setState({ grown: true })));
  }

  async load() {
    const [boot, view] = await Promise.all([get('/api/bootstrap'), get('/api/view')]);
    this.setState({ boot, view });
  }

  go(id) { return () => this.setState({ screen: id, fabOpen: false }); }
  setForm(p) { this.setState(s => ({ form: Object.assign({}, s.form, p) })); }

  async analyse() {
    this.setState({ busy: true });
    const body = Object.assign({}, this.state.form, {
      preset: this.state.preset || null,
      reportJson: this.state.upload ? this.state.upload.report : null
    });
    const view = await post('/api/analyse', body);
    const boot = await get('/api/bootstrap');
    this.setState({ view, boot, busy: false, screen: 'assessment',
                    messages: this.state.messages.slice(0, 1) });
  }

  async loadPreset(key) {
    // Analysed server-side as the CANONICAL record. Posting a partial form produced an
    // encounter with one citable fact and thirteen absent values — a different record from the
    // scenario it was named after, whose recorded differential was then correctly refused.
    this.setState({ busy: true });
    const view = await get('/api/preset?key=' + encodeURIComponent(key));
    const boot = await get('/api/bootstrap');
    this.setState({ preset: key, view, boot, busy: false, screen: 'assessment',
                    messages: this.state.messages.slice(0, 1) });
  }

  async upload(files) {
    if (!files || !files.length) return;
    const reads = [].slice.call(files).slice(0, 2).map(f => new Promise(res => {
      const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(f);
    }));
    const imgs = await Promise.all(reads);
    this.setState({ busy: true, previews: imgs });
    const out = await post('/api/upload', { organ: this.state.form.organ, image: imgs[0],
                                            image2: imgs[1] || null });
    this.setState({ upload: out, busy: false });
  }

  async ask(q) {
    const text = (q || '').trim();
    if (!text) return;
    this.setState(s => ({ draft: '', busy: true,
                          messages: s.messages.concat([{ role: 'd', text }]) }));
    const out = await post('/api/ask', { question: text });
    this.setState(s => ({ busy: false,
                          messages: s.messages.concat([{ role: 'a', text: out.answer }]) }));
  }

  navStyle(id, off) {
    const on = this.state.screen === id;
    return { appearance: 'none', border: 0, cursor: off ? 'not-allowed' : 'pointer',
      textAlign: 'left', width: '100%', display: 'flex', alignItems: 'center',
      justifyContent: 'space-between', gap: '10px', padding: '12px 16px', borderRadius: '8px',
      background: on ? '#E9E8FB' : 'transparent',
      color: on ? '#2E2A78' : (off ? '#B9B5EC' : '#6A6785'),
      fontWeight: on ? 700 : 500, fontSize: '14.5px',
      boxShadow: on ? 'inset 3px 0 0 #5B54D6' : 'none', opacity: off ? 0.55 : 1 };
  }

  flagStyle(flag) {
    const bad = flag && flag !== 'normal' && flag !== '';
    return { fontSize: '12px', fontWeight: 700,
             color: !flag ? '#9C99B8' : bad ? '#C13238' : '#5A7A0F' };
  }

  renderVals() {
    const st = this.state, v = st.view || {}, boot = st.boot || {};
    const has = !!v.hasEncounter;
    const sev = v.severity || '—';
    const t = sev === 'HIGH' ? { bg: '#FDECEC', fg: '#C13238' }
            : sev === 'MODERATE' ? { bg: '#FFF3E0', fg: '#9A6207' }
            : { bg: '#F0FADB', fg: '#5A7A0F' };
    const nAlerts = (v.alerts || []).length;
    const f = st.form;

    // How much of the workup has been filled — a real count, not a fixed "3 of 5".
    const filled = [
      !!(f.name || f.complaint), Object.keys(f.vitals).length > 0,
      Object.keys(f.labs).length > 0,
      !!st.upload || Object.keys(f.findings).length > 0, has
    ].filter(Boolean).length;

    const out = {
      ready: !!st.boot, busy: st.busy, hasEncounter: has, noEncounter: !has,
      analyzing: st.busy, analyzed: has,
      analyzeLabel: st.busy ? 'Analyzing…' : (has ? 'Re-analyze patient' : 'Analyze patient'),
      filledCount: String(filled),
      onAnalyze: () => this.analyse(),
      emptyMessage: v.message || 'No encounter has been analysed yet.',

      nav: SCREENS.map(s => {
        const off = NEEDS.indexOf(s.id) >= 0 && !has;
        const badge = s.id === 'alerts' && has && nAlerts ? String(nAlerts) : (s.badge || '');
        return { label: s.label, icon: s.icon, badge: badge,
          style: this.navStyle(s.id, off), onClick: off ? (() => {}) : this.go(s.id),
          badgeStyle: badge ? { background: s.id === 'alerts' ? '#FDECEC' : '#E4E2F8',
            color: s.id === 'alerts' ? '#C13238' : '#5B3CC4', borderRadius: '999px',
            padding: '3px 9px', fontSize: '11.5px', fontWeight: 700 } : { display: 'none' } };
      }),

      // ---- patient ---------------------------------------------------------------
      pName: (v.patient || {}).name || 'No patient',
      pAge: (v.patient || {}).age || '—',
      pSex: (v.patient || {}).sex || '',
      pComplaint: (v.patient || {}).complaint || '—',
      pOrgan: (v.patient || {}).organ || '—',
      pId: (v.patient || {}).id || '—',
      severity: sev, scenario: v.scenario || '—',
      thresholdsVersion: v.thresholdsVersion || '—',
      conclusion: v.conclusion || '',
      escalateText: v.escalate ? 'escalation required' : 'no escalation',
      caseQuality: v.caseQuality || '—',
      alertCount: String(nAlerts),
      missingCount: String((v.missing || []).length),
      severityStyle: { background: t.bg, color: t.fg, borderRadius: '999px',
                       padding: '5px 12px', fontSize: '12.5px', fontWeight: 700 },
      severityFg: { color: t.fg, fontSize: '12.5px', letterSpacing: '.11em',
                    textTransform: 'uppercase', fontWeight: 800 },
      triggers: (v.triggers || []).map(x => ({ text: x })),
      missing: (v.missing || []).map(m => ({ name: m })),
      outOfScope: (v.outOfScope || []).map(x => ({ text: x })),
      notAssessed: (v.notAssessed || []).map(o => ({ organ: o })),

      // ---- workup: the form fields the design lays out ---------------------------
      fName: f.name, fAge: String(f.age), fSex: f.sex, fComplaint: f.complaint,
      fHistory: f.history, fOrgan: f.organ,
      onName: e => this.setForm({ name: e.target.value }),
      onAge: e => this.setForm({ age: parseInt(e.target.value || '0', 10) || 0 }),
      onSex: e => this.setForm({ sex: e.target.value === 'Male' ? 'M' : 'F' }),
      onComplaint: e => this.setForm({ complaint: e.target.value }),
      onHistory: e => this.setForm({ history: e.target.value }),
      onOrgan: e => this.setForm({ organ: e.target.value }),
      onUpload: e => this.upload(e.target.files),
      clips: (st.previews || []).map(src => ({ src })),
      uploadLabel: st.busy ? 'Reading…'
                 : (st.previews || []).length ? '＋ Replace'
                 : (f.organ === 'Heart' ? '＋ Add ED + ES' : '＋ Add clip'),

      // The organs this deployment can actually run, from module_status. The mockup offered
      // FAST, for which no module was ever built; an examination tab with nothing behind it is
      // a claim that cannot be backed. A module whose weights are missing is shown, but
      // disabled and labelled, because that is a different fact from the organ not existing.
      organChips: Object.keys(boot.modules || {}).map(k => {
        const m = boot.modules[k];
        const on = f.organ.toLowerCase() === k;
        // The schema calls it `heart`; clinicians and the design call the study cardiac.
        const name = k === 'heart' ? 'Cardiac'
                   : k.charAt(0).toUpperCase() + k.slice(1);
        return {
          label: name + (m.runs ? '' : ' · unavailable'),
          onClick: m.runs
            ? () => this.setForm({ organ: k.charAt(0).toUpperCase() + k.slice(1) })
            : (() => {}),
          style: { appearance: 'none', border: 0, borderRadius: '999px', padding: '7px 14px',
            fontSize: '12.5px', fontWeight: on ? 700 : 600,
            cursor: m.runs ? 'pointer' : 'not-allowed',
            background: on ? '#5B54D6' : '#EFEEFB',
            color: on ? '#fff' : (m.runs ? '#6A6785' : '#B9B5EC'),
            opacity: m.runs ? 1 : 0.6 }
        };
      }),

      vitalFields: (boot.vitalKeys || []).map(k => {
        const val = f.vitals[k.key];
        const flag = val === undefined ? 'not measured'
          : (val < k.min ? 'low' : val > k.max ? 'high' : 'normal');
        return { label: k.key, unit: k.unit, value: val === undefined ? '' : String(val),
          flag: flag, flagStyle: this.flagStyle(val === undefined ? '' : flag),
          inputStyle: { border: '1px solid #DEDCF4', borderRadius: '10px',
                        padding: '10px 12px', fontSize: '14.5px', background: '#FCFBFF',
                        width: '100%' },
          onChange: e => {
            const x = e.target.value, nv = Object.assign({}, this.state.form.vitals);
            if (x === '') delete nv[k.key]; else nv[k.key] = parseFloat(x);
            this.setForm({ vitals: nv });
          } };
      }),

      labFields: (boot.labKeys || []).map(k => {
        const val = f.labs[k.key];
        const flag = val === undefined ? 'not measured'
          : (k.max !== null && val > k.max ? 'high' : 'normal');
        return { name: k.key, value: val === undefined ? '' : String(val), flag: flag,
          nameStyle: { fontWeight: 600, fontSize: '14px',
                       color: val === undefined ? '#9C99B8' : '#1B1A3A' },
          flagStyle: this.flagStyle(val === undefined ? '' : flag),
          inputStyle: { border: '1px solid #DEDCF4', borderRadius: '10px',
                        padding: '9px 12px', fontSize: '14px', background: '#FCFBFF',
                        width: '100%' },
          onChange: e => {
            const x = e.target.value, nl = Object.assign({}, this.state.form.labs);
            if (x === '') delete nl[k.key]; else nl[k.key] = parseFloat(x);
            this.setForm({ labs: nl });
          } };
      }),

      // What the module reported, once it has read an image. Before that, the row shows the
      // finding names it screens for and says nothing has been read.
      pocusFields: (st.upload ? st.upload.rows : (boot.lungFindings || []).map(n => ({
          label: n.replace(/_/g, ' '), detected: false, conf: null, caption: '' })))
        .map(r => ({
          name: r.label,
          value: r.conf === null || r.conf === undefined ? 'not read yet'
                 : (r.detected ? 'Detected ' + r.conf : 'Not detected ' + r.conf),
          nameStyle: { fontWeight: 600, fontSize: '14px' },
          segStyle: { fontSize: '12.5px', fontWeight: 700,
                      color: r.conf === null || r.conf === undefined ? '#9C99B8'
                             : r.detected ? '#2E2A78' : '#6A6785' } })),

      zones: (v.findings || []).slice(0, 8).map((x, i) => ({
        label: x.label.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 3),
        style: { aspectRatio: '1', borderRadius: '6px', display: 'flex',
          alignItems: 'center', justifyContent: 'center', fontWeight: 800,
          fontSize: '13.5px', background: x.detected ? '#5B54D6' : '#C6F24E',
          color: x.detected ? '#fff' : '#1B1A3A',
          animation: 'riseIn .45s ' + (i * 0.05).toFixed(2) + 's cubic-bezier(.2,.7,.3,1) both' }
      })),

      steps: (v.timeline || []).map(s => ({
        label: s.title + ' — ' + s.detail,
        dotStyle: { width: '9px', height: '9px', borderRadius: '50%',
                    background: s.hl ? '#5B54D6' : '#C6F24E', display: 'block' },
        textStyle: { fontSize: '13.5px', color: '#6A6785' } })),

      // ---- imaging ---------------------------------------------------------------
      findings: (v.findings || []).map(x => ({
        label: x.label, caption: x.caption, conf: x.conf.toFixed(2),
        status: x.detected ? 'Detected' : 'Not detected',
        statusStyle: x.detected
          ? { background: '#E4E2F8', color: '#2E2A78', borderRadius: '999px',
              padding: '5px 12px', fontSize: '12.5px', fontWeight: 700 }
          : { color: '#6A6785', fontSize: '13.5px' } })),
      topFinding: v.topFinding || '—',
      hasUpload: !!st.upload,
      uploadSeconds: st.upload ? String(st.upload.seconds) : '',
      uploadModel: st.upload ? (st.upload.report.model || 'the module') : '',
      uploadScope: st.upload ? ((st.upload.report.reliability || {}).scope || '') : '',

      // ---- vitals & labs as displayed elsewhere ----------------------------------
      vitals: (v.vitals || []).map(x => ({
        label: x.label, unit: x.unit,
        value: x.value === null ? 'Not measured' : x.value,
        note: x.value === null ? 'Not the same as normal' : (x.flag || ''),
        flag: x.flag || 'not measured', flagStyle: this.flagStyle(x.flag),
        style: x.value === null
          ? { background: '#FAFAFE', border: '1px dashed #DCD4F7', borderRadius: '18px',
              padding: '20px 22px' }
          : { background: '#fff', border: '1px solid #E4E2F8', borderRadius: '18px',
              padding: '20px 22px',
              borderTop: '4px solid ' + (x.flag === 'normal' ? '#22A06B' : '#E5484D') },
        valueStyle: { fontSize: x.value === null ? '20px' : '32px', fontWeight: 800,
                      letterSpacing: '-.02em',
                      color: x.value === null ? '#9C99B8' : '#1B1A3A' } })),
      labs: (v.labs || []).map(x => ({
        name: x.name, result: x.result || '—', ref: x.ref,
        status: x.result === null ? 'Not measured' : (x.flag || ''),
        rowStyle: { padding: '13px 0', color: x.result === null ? '#9C99B8' : '#1B1A3A' },
        statusStyle: { borderRadius: '999px', padding: '4px 11px', fontSize: '12.5px',
          fontWeight: 700,
          background: x.result === null ? '#EFEEFB' : x.flag === 'normal' ? '#F0FADB' : '#FDECEC',
          color: x.result === null ? '#5B3CC4' : x.flag === 'normal' ? '#5A7A0F' : '#C13238' } })),

      // ---- the ranked differential -----------------------------------------------
      ranked: (v.differential || []).map((d, i) => ({
        rank: String(i + 1).padStart(2, '0'), name: d.diagnosis, tag: d.likelihood,
        supports: (d.supporting || []).join('; ') || '—',
        limits: ((d.limitations || []).concat(d.contradicting || [])).join('; ') || '—',
        now: (d.supportingIds || []).join(', ') || '—',
        titleStyle: { fontSize: i === 0 ? '22px' : '19px', fontWeight: i === 0 ? 800 : 700 },
        cardStyle: i === 0
          ? { border: '1px solid #DEDCF4', borderRadius: '16px', padding: '20px 22px',
              background: '#FCFBFF', marginBottom: '16px' }
          : { border: '1px solid #E4E2F8', borderRadius: '16px', padding: '20px 22px',
              marginBottom: '16px' },
        fillStyle: { height: '100%',
          width: (d.supportingIds || []).length
                 ? Math.min(100, (d.supportingIds || []).length * 25) + '%' : '6%',
          background: 'linear-gradient(90deg,#5B54D6,#2E2A78)' },
        tagStyle: { borderRadius: '999px', padding: '6px 14px', fontSize: '13px',
                    fontWeight: 700, background: '#E4E2F8', color: '#2E2A78' } })),
      hasDifferential: (v.differential || []).length > 0,
      noDifferential: (v.differential || []).length === 0,
      differentialNote: v.differentialOrigin === 'failed'
        ? 'The model backend failed and the answer was withheld. The severity and alerts beside '
          + 'it were computed before the model ran and are unaffected.'
        : v.differentialOrigin === 'not_generated'
        ? 'No differential was generated. Producing one requires a 4.9 GB language model on a '
          + 'GPU, which is not loaded in this deployment. Everything else on screen was '
          + 'computed here.' : '',
      withheld: !!v.withheld,
      validationErrors: (v.validationErrors || []).map(x => ({ text: x })),
      warnings: (v.warnings || []).map(x => ({ text: x })),
      evidence: (v.evidence || []).map(e => ({ id: e.id, text: e.text,
        mark: e.cited ? '●' : '○', style: { color: e.cited ? '#1B1A3A' : '#6A6785' } })),
      hits: (v.hits || []).map(h => ({ n: String(h.n), id: h.id, topic: h.topic,
        score: h.score.toFixed(2), source: h.source, text: h.text })),
      noHits: (v.hits || []).length === 0,

      // ---- alerts ------------------------------------------------------------------
      alerts: (v.alerts || []).map(a => {
        const crit = a.severity === 'CRITICAL';
        return { type: a.type, message: a.message,
          kicker: crit ? 'Immediate attention' : 'Important',
          style: { background: '#fff', borderRadius: '18px', padding: '22px 26px',
            marginBottom: '16px', border: '1px solid ' + (crit ? '#F6CFCF' : '#F7DFB4'),
            borderLeft: '5px solid ' + (crit ? '#E5484D' : '#F5A623') },
          kickerStyle: { color: crit ? '#C13238' : '#9A6207', fontSize: '12.5px',
            letterSpacing: '.11em', textTransform: 'uppercase', fontWeight: 800 } };
      }),
      noAlerts: (v.alerts || []).length === 0,

      exams: (v.exams || []).map(e => ({ exam: e.exam, reason: e.reason,
                                         priority: e.priority })),
      nextStep: (v.exams || []).length ? v.exams[0].exam
                : 'Physician review of the current findings.',
      therapeutic: ((v.therapeutic || {}).considerations || []).map(c => ({
        consideration: c.consideration, basis: c.basis, passage: c.passage,
        disclaimer: c.disclaimer })),
      noTherapeutic: !((v.therapeutic || {}).considerations || []).length,
      therapeuticStatus: (v.therapeutic || {}).status || '',

      // ---- charts: counts of cited evidence, never probabilities -------------------
      bars: (v.bars || []).map(b => {
        const tot = Math.max(b.sup + b.ag, 1);
        const split = Math.round(b.sup / tot * 100);
        const mx = Math.max.apply(null,
          (v.bars || []).map(x => Math.max(x.sup, x.ag)).concat([1]));
        return { label: b.label, barStyle: { width: '100%', maxWidth: '46px',
          height: st.grown ? Math.round(Math.max(b.sup, b.ag) / mx * 100) + '%' : '0%',
          transition: 'height .8s cubic-bezier(.2,.8,.25,1)',
          background: 'linear-gradient(180deg,#5B54D6 0 ' + split + '%,#2E2A78 '
                      + split + '% 100%)' } };
      }),
      heat: (v.evidence || []).slice(0, 28).map(e => ({
        style: { display: 'block', aspectRatio: '1', borderRadius: '2px',
                 background: e.cited ? '#5B54D6'
                   : /HIGH|LOW/.test(e.text) ? '#F4AFB2' : '#E4E2F8' } })),

      timeline: (v.timeline || []).map(x => ({ time: x.time, title: x.title,
        detail: x.detail,
        dotStyle: { width: '13px', height: '13px', borderRadius: '50%', marginTop: '5px',
          background: x.hl ? '#5B3CC4' : '#fff',
          border: x.hl ? '3px solid #D9CDFF' : '3px solid #DEDCF4', display: 'block' },
        titleStyle: { fontWeight: x.hl ? 800 : 600, fontSize: '15.5px',
                      color: x.hl ? '#2E2A78' : '#1B1A3A' } })),

      reportText: v.reportText || '', generatedAt: v.generatedAt || '',

      // ---- roster / session record -------------------------------------------------
      roster: (boot.roster || []).map(r => ({ name: r.name, age: String(r.age), sex: r.sex,
        complaint: r.complaint, tag: r.tag, severity: r.severity,
        alerts: r.alerts + ' alert(s)',
        initials: r.name.split(' ').map(w => w[0]).join('').toUpperCase(),
        onOpen: () => this.loadPreset(r.key),
        tagStyle: { borderRadius: '999px', padding: '5px 12px', fontSize: '12.5px',
          fontWeight: 700,
          background: r.tag === 'Critical' ? '#FDECEC'
                    : r.tag === 'Review' ? '#FFF3E0' : '#F0FADB',
          color: r.tag === 'Critical' ? '#C13238'
               : r.tag === 'Review' ? '#9A6207' : '#5A7A0F' } })),
      visits: (boot.records || []).map(r => ({ date: r.at, reason: r.name,
        meta: r.organ + ' · ' + r.alerts + ' alert(s)', tag: r.severity,
        outcome: (r.findings || []).join(', ') || 'no positive finding',
        images: [], onOpen: this.go('report'),
        tagStyle: { borderRadius: '999px', padding: '5px 12px', fontSize: '12.5px',
          fontWeight: 700, background: r.severity === 'HIGH' ? '#FDECEC' : '#F0FADB',
          color: r.severity === 'HIGH' ? '#C13238' : '#5A7A0F' } })),
      noRecords: !(boot.records || []).length,
      dataRows: (v.vitals || []).concat(v.labs || []).map(x => ({
        name: x.label || x.name,
        now: x.value === null || x.result === null ? 'Not measured'
             : String(x.value !== undefined ? x.value : x.result),
        v2: '—', v3: '—', v4: '—',
        nowStyle: { padding: '13px 0', fontWeight: 700,
          color: (x.flag && x.flag !== 'normal') ? '#C13238' : '#1B1A3A' } })),
      rosterCount: String((boot.roster || []).length),
      sessionCount: String((boot.records || []).length),
      criticalCount: String((boot.roster || []).filter(r => r.severity === 'HIGH').length),
      testCount: boot.tests ? String(boot.tests) : '—',
      modules: Object.keys(boot.modules || {}).map(k => ({ organ: k,
        reason: boot.modules[k].reason, dot: boot.modules[k].runs ? '●' : '○' })),

      // ---- assistant ---------------------------------------------------------------
      messages: st.messages.map(m => ({ text: m.text,
        rowStyle: { display: 'flex',
                    justifyContent: m.role === 'd' ? 'flex-end' : 'flex-start' },
        bubbleStyle: m.role === 'd'
          ? { maxWidth: '74%', background: '#5B54D6', color: '#fff', padding: '14px 18px',
              borderRadius: '16px 16px 4px 16px', fontSize: '14.5px', lineHeight: 1.55,
              whiteSpace: 'pre-wrap' }
          : { maxWidth: '78%', background: '#F4F3FD', border: '1px solid #E4E2F8',
              color: '#1B1A3A', padding: '14px 18px', borderRadius: '16px 16px 16px 4px',
              fontSize: '14.5px', lineHeight: 1.6, whiteSpace: 'pre-wrap' },
        miniBubbleStyle: m.role === 'd'
          ? { alignSelf: 'flex-end', background: '#5B54D6', color: '#fff',
              padding: '9px 12px', borderRadius: '12px 12px 3px 12px', fontSize: '13px',
              maxWidth: '85%', whiteSpace: 'pre-wrap' }
          : { alignSelf: 'flex-start', background: '#F4F3FD', border: '1px solid #E4E2F8',
              color: '#1B1A3A', padding: '9px 12px', borderRadius: '12px 12px 12px 3px',
              fontSize: '13px', maxWidth: '90%', whiteSpace: 'pre-wrap' } })),
      suggestions: SUGGESTIONS.map(label => ({ label, onClick: () => this.ask(label) })),
      draft: st.draft,
      onDraft: e => this.setState({ draft: e.target.value }),
      onSubmit: e => { e.preventDefault(); this.ask(this.state.draft); },
      askWhy: () => this.ask('Why is this the severity?'),
      askChallenge: () => this.ask('Challenge this assessment.'),
      askBLines: () => this.ask('What did POCUS actually see?'),

      // ---- floating assistant button -----------------------------------------------
      fabOpen: st.fabOpen,
      fabBadge: has && nAlerts > 0,
      fabIcon: st.fabOpen ? '×' : '✧',
      toggleFab: () => this.setState(s => ({ fabOpen: !s.fabOpen })),
      fabStyle: { position: 'fixed', right: '26px', bottom: '26px', width: '56px',
        height: '56px', borderRadius: '50%', border: 0, cursor: 'pointer',
        background: '#5B54D6', color: '#fff', fontSize: '22px', zIndex: 40,
        boxShadow: '0 10px 26px rgba(46,42,120,.34)' },

      recordTabs: [['images', 'POCUS images'], ['data', 'Measurements'],
                   ['visits', 'Encounters'], ['reports', 'Reports']].map(([id, label]) => ({
        label, onClick: () => this.setState({ recTab: id }),
        style: { appearance: 'none', cursor: 'pointer', borderRadius: '8px',
          padding: '11px 18px', fontWeight: 700, fontSize: '14px',
          background: st.recTab === id ? '#5B54D6' : '#fff',
          color: st.recTab === id ? '#fff' : '#6A6785',
          border: '1px solid ' + (st.recTab === id ? '#5B54D6' : '#DEDCF4') } })),
      recImages: st.recTab === 'images', recData: st.recTab === 'data',
      recVisits: st.recTab === 'visits', recReports: st.recTab === 'reports'
    };

    SCREENS.forEach(s => {
      const cap = s.id[0].toUpperCase() + s.id.slice(1);
      out['on' + cap] = st.screen === s.id;
      out['go' + cap] = this.go(s.id);
    });
    return out;
  }
}
