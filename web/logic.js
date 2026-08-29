// Replaces the mockup's constants. Everything clinical comes from /api/*; only nav labels and
// empty-state copy live here, because those are interface chrome rather than patient data.

// The language is stamped into the page at bind time, so the interface is chosen before any
// script runs rather than swapped afterwards. `T` is used only for chrome written in this
// file; every clinical sentence is rendered in French by src/agents/i18n.py from the same
// computed fields, so it cannot drift from the English record.
const LANG = (typeof window !== 'undefined' && window.LANG) || 'en';
const FR = LANG === 'fr';
const T = (en, fr) => (FR ? fr : en);

const SCREENS = [
  { id: 'home', label: T('Home', 'Accueil'), icon: '⌂' },
  { id: 'workup', label: T('Patient workup', 'Saisie du patient'), icon: '✎' },
  { id: 'diagnosis', label: T('Diagnosis', 'Diagnostic'), icon: '✳' },
  { id: 'record', label: T('Patient record', 'Dossier patient'), icon: '▣' },
  { id: 'assessment', label: T('Assessment', 'Évaluation'), icon: '✦', badge: 'IA' },
  { id: 'alerts', label: T('Alerts', 'Alertes'), icon: '⚠' },
  { id: 'assistant', label: T('Clinical assistant', 'Assistant clinique'), icon: '✧' },
  { id: 'timeline', label: T('Timeline', 'Chronologie'), icon: '◷' },
  { id: 'report', label: T('Report', 'Rapport'), icon: '▤' },
  { id: 'history', label: T('History', 'Historique'), icon: '⟲' }
];

// Screens that read a computed encounter. Without one they have nothing to show, so the nav
// marks them rather than letting a click land on an empty page and read as a broken button.
const NEEDS = ['diagnosis', 'assessment', 'alerts', 'assistant', 'timeline', 'report'];

const SUGGESTIONS = FR ? [
  'Quels signes soutiennent la principale hypothèse ?',
  'Quelles informations manquent ?',
  'Qu’a réellement vu l’échographie ?',
  'Résume ce patient.',
  'Que dois-je faire maintenant ?',
  'Conteste cette évaluation.'
] : [
  'What findings support the leading entry?',
  'What information is missing?',
  'What did POCUS actually see?',
  'Summarise this patient.',
  'What should I investigate next?',
  'Challenge this assessment.'
];

// The language rides on every request, so the server renders the clinical text in the same
// language as the page that asked for it.
const _lang = u => u + (u.indexOf('?') < 0 ? '?' : '&') + 'lang=' + LANG;
const post = (u, b) => fetch(_lang(u), { method: 'POST',
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })
  .then(r => r.json());
const get = u => fetch(_lang(u)).then(r => r.json());

class Component extends DCLogic {
  state = {
    screen: 'home', recTab: 'images', draft: '', busy: false, grown: false, fabOpen: false,
    boot: null, view: null, preset: '', upload: null, previews: [],
    form: { name: '', age: 60, sex: 'F', complaint: '', history: '', tier: 'medium',
            tconf: 0.8, organ: 'Lung', vitals: {}, labs: {}, findings: {} },
    // What was TYPED, kept beside what was parsed. The box showed the parsed number, so
    // "20." came back as "20" and the decimal point could never be entered at all.
    typed: { vitals: {}, labs: {} },
    messages: [{ role: 'a', text: T(
      'Ask me anything about this encounter. I answer from the computed assessment — the '
      + 'clinical state, the escalation triggers, the alerts, the evidence and the '
      + 'differential as validated.',
      'Posez-moi n’importe quelle question sur cette prise en charge. Je réponds à partir de '
      + 'l’évaluation calculée — l’état clinique, les déclencheurs d’escalade, les alertes, '
      + 'les éléments de preuve et le diagnostic différentiel tel que validé.') }]
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

  // Editing the form makes this a NEW encounter. The preset must be cleared with it, or the
  // next analyse still carries the benchmark key and the server compares the typed patient
  // against a scenario it is no longer describing.
  setForm(p) {
    this.setState(s => ({ form: Object.assign({}, s.form, p), preset: '' }));
  }

  // parseFloat('') and parseFloat('abc') are both NaN, and JSON.stringify writes NaN as null.
  // The API rejects null for a float, so a single unparseable box silently failed the whole
  // analysis and left the previous encounter on screen. A value that is not a number is not a
  // measurement: the key is removed, which records it as never measured.
  static num(x) {
    if (x === '' || x === null || x === undefined) return null;
    // A comma is a decimal separator to most of the people this is for, and the interface
    // offers French. parseFloat('20,4') is 20 -- it would silently drop the fraction.
    const n = parseFloat(String(x).replace(',', '.'));
    return isFinite(n) ? n : null;
  }

  // The box shows what was typed; the encounter carries what parsed. Binding the box to the
  // parsed number meant every keystroke was re-rendered through parseFloat, so "20." became
  // "20" and the point was erased as it was typed: no decimal could be entered, and 36.6 C or
  // a lactate of 2.6 could not be recorded at all. They are separate now, and a box holding
  // "20." is simply not a measurement yet -- the key stays absent until it parses.
  setNumeric(group, key, raw) {
    const next = Object.assign({}, this.state.form[group]);
    const n = Component.num(raw);
    if (n === null) delete next[key]; else next[key] = n;
    const patch = {}; patch[group] = next;
    this.setState(s => {
      const t = Object.assign({}, s.typed);
      t[group] = Object.assign({}, t[group]);
      if (raw === '') delete t[group][key]; else t[group][key] = raw;
      return { typed: t };
    });
    this.setForm(patch);
  }

  async analyse() {
    this.setState({ busy: true, error: '' });
    // Wait for a study still being read. Pressing Analyze while the upload was in flight took
    // `upload` as null and sent BOTH reportJson and images empty -- so the encounter was
    // assessed from whatever was left in the form, reported findings that did not come from
    // the scan just uploaded, and filed no image against the patient. It looked like the
    // upload had been ignored; it had in fact been overtaken. The first read after startup
    // loads the weights and takes seconds, which is exactly when a clinician clicks on.
    if (this._reading) { try { await this._reading; } catch (e) { /* reported below */ } }
    const body = Object.assign({}, this.state.form, {
      preset: this.state.preset || null,
      reportJson: this.state.upload ? this.state.upload.report : null,
      // The studies this encounter was read from, so they are filed under the patient rather
      // than discarded once the assessment is made.
      images: this.state.upload ? (this.state.upload.stored || []) : []
    });
    const view = await post('/api/analyse', body);
    // A rejected request used to leave the previous encounter on screen, so analysing looked
    // like it had analysed somebody else. Say so instead.
    if (!view || !view.hasEncounter) {
      this.setState({ busy: false,
        error: T('That encounter could not be analysed: ',
                 'Cette prise en charge n’a pas pu être analysée : ')
               + (view && view.detail ? JSON.stringify(view.detail)
                  : T('the server rejected it', 'le serveur l’a rejetée'))
               + T('. Nothing on screen has changed.',
                   '. Rien n’a changé à l’écran.') });
      return;
    }
    const boot = await get('/api/bootstrap');
    // Point the record at the patient just assessed. It was left on whoever was opened last,
    // so Patient record drew this patient's header -- name, complaint, alerts, severity, all
    // from the new encounter -- while the images tab filtered by the OLD selection. Assessing
    // a patient whose predecessor had no study showed "no study has been read for this
    // patient" above a counter saying two had been, and the image really was stored.
    const rs = boot.records || [];
    this.setState({ view, boot, busy: false, error: '', screen: 'assessment',
                    recordSel: rs.length ? rs[rs.length - 1].id : null,
                    messages: this.state.messages.slice(0, 1) });
  }

  // The patient record is a LIST first. A clinician opens a patient, and only then sees that
  // patient's case, history, diagnostics and timeline. `recordSel` null means the list.
  openRecord(id) {
    return async () => {
      this.setState({ busy: true });
      const view = await get('/api/record?id=' + encodeURIComponent(id));
      // Also opened from the home table, so it carries the screen with it.
      this.setState({ view, recordSel: id, busy: false, recTab: 'images', screen: 'record' });
    };
  }

  backToList() { this.setState({ recordSel: null }); }

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

  // Every file, not the first two. Several files are several studies unless the clinician
  // says they are frames of one acquisition.
  readFiles(files) {
    return Promise.all([].slice.call(files).slice(0, 8).map(f => new Promise(res => {
      const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(f);
    })));
  }

  // Filing a study against a patient already assessed. The reading is shown with the image;
  // the assessment is NOT re-run, because it was reached without this scan.
  async addToRecord(files) {
    if (!files || !files.length || !this.state.recordSel) return;
    const rec = ((this.state.boot || {}).records || [])
      .filter(r => r.id === this.state.recordSel)[0];
    const imgs = await this.readFiles(files);
    this.setState({ busy: true });
    const up = await post('/api/upload', { organ: (rec && rec.organ) || 'Lung',
                                           image: imgs[0], images: imgs, asClip: false });
    const view = await post('/api/record/attach', { id: this.state.recordSel,
                                                    images: up.stored || [] });
    const boot = await get('/api/bootstrap');
    this.setState({ view, boot, busy: false });
  }

  async upload(files) {
    if (!files || !files.length) return;
    // Held so that analyse() can wait on it rather than race it.
    this._reading = this._read(files);
    try { await this._reading; } finally { this._reading = null; }
  }

  async _read(files) {
    const imgs = await this.readFiles(files);
    this.setState({ busy: true, previews: imgs });
    const out = await post('/api/upload', {
      organ: this.state.form.organ, image: imgs[0], images: imgs,
      image2: imgs[1] || null, asClip: !!this.state.asClip
    });
    this.setState({ upload: out, busy: false });
  }

  async toggleClip() {
    const next = !this.state.asClip;
    this.setState({ asClip: next });
    if (this.state.previews && this.state.previews.length > 1) {
      // Re-read, and held like any other read so analyse() waits for it.
      this._reading = (async () => {
        this.setState({ busy: true });
        const out = await post('/api/upload', {
          organ: this.state.form.organ, image: this.state.previews[0],
          images: this.state.previews, image2: this.state.previews[1] || null, asClip: next
        });
        this.setState({ upload: out, busy: false });
      })();
      try { await this._reading; } finally { this._reading = null; }
    }
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
    // Compared on the untranslated key. Matching the DISPLAYED word meant a French screen
    // painted every severity the same colour, because "ÉLEVÉE" is not "HIGH" -- a comparison
    // that breaks in one language and not the other, silently, in the colour a clinician
    // reads before any word.
    const sevKey = v.severityKey || v.severity || '—';
    const t = sevKey === 'HIGH' ? { bg: '#FDECEC', fg: '#C13238' }
            : sevKey === 'MODERATE' ? { bg: '#FFF3E0', fg: '#9A6207' }
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
      analyzeLabel: st.busy ? T('Analyzing…', 'Analyse en cours…') : (has ? T('Re-analyze patient', 'Réanalyser le patient') : T('Analyze patient', 'Analyser le patient')),
      filledCount: String(filled),
      onAnalyze: () => this.analyse(),
      analyseLabel: st.busy ? T('Reading the study…', 'Lecture de l’examen…') : T('✦ Analyze patient', '✦ Analyser le patient'),
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
      escalateText: v.escalate ? T('escalation required', 'escalade requise') : T('no escalation', 'pas d’escalade'),
      caseQuality: v.caseQuality || '—',
      alertCount: String(nAlerts),
      alertHeadline: nAlerts
        ? nAlerts + T(' alert(s) require physician attention',
                      ' alerte(s) nécessitent l’attention du médecin')
        : T('No alert was raised for this encounter.',
            'Aucune alerte n’a été déclenchée pour cette prise en charge.'),
      // On the workup this counts what the CLINICIAN has left blank, which is the number the
      // note beside it is about. It previously read from the last analysis, so a fresh form
      // always claimed nothing was missing.
      missingCount: String(has
        ? (v.missing || []).length
        : (boot.vitalKeys || []).filter(k => f.vitals[k.key] === undefined).length
          + (boot.labKeys || []).filter(k => f.labs[k.key] === undefined).length),
      error: st.error || '',
      hasError: !!st.error,
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
      // A placeholder, not content: the box used to arrive holding a fabricated history.
      historyPlaceholder: T('e.g. hypertension, prior heart failure; current medications',
                            'ex. hypertension, insuffisance cardiaque ; traitements en cours'),
      onOrgan: e => this.setForm({ organ: e.target.value }),
      onUpload: e => this.upload(e.target.files),
      clips: (st.previews || []).map((src, i) => ({ src, n: String(i + 1) })),
      uploadNote: st.upload ? (st.upload.note || '') : '',
      uploadCount: st.upload ? String(st.upload.count || 1) : '0',
      multiUpload: !!(st.upload && st.upload.perImage && st.upload.perImage.length),
      perImage: ((st.upload || {}).perImage || []).map(p => ({
        title: 'Study ' + p.index + (p.status === 'ok' ? '' : ' — ' + p.status),
        rows: p.rows.map(r => ({ label: r.label, caption: r.caption,
          conf: r.conf.toFixed(2), status: r.detected ? T('Detected', 'Détecté') : T('Not detected', 'Non détecté') })) })),
      asClip: !!st.asClip,
      clipToggleLabel: st.asClip ? T('Frames of one clip', 'Images d’une même boucle') : T('Separate studies', 'Examens distincts'),
      onToggleClip: () => this.toggleClip(),
      uploadLabel: st.busy ? 'Reading…'
                 : (st.previews || []).length ? T('＋ Replace', '＋ Remplacer')
                 : (f.organ === 'Heart' ? T('＋ Add ED + ES', '＋ Ajouter TD + TS') : T('＋ Add clip', '＋ Ajouter une boucle')),

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
          label: name + (m.runs ? '' : T(' · unavailable', ' · indisponible')),
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
        const typed = (st.typed.vitals || {})[k.key];
        return { label: k.key, unit: k.unit,
          value: typed !== undefined ? typed : (val === undefined ? '' : String(val)),
          flag: flag, flagStyle: this.flagStyle(val === undefined ? '' : flag),
          inputStyle: { border: '1px solid #DEDCF4', borderRadius: '10px',
                        padding: '10px 12px', fontSize: '14.5px', background: '#FCFBFF',
                        width: '100%' },
          onChange: e => this.setNumeric('vitals', k.key, e.target.value) };
      }),

      labFields: (boot.labKeys || []).map(k => {
        const val = f.labs[k.key];
        // Both directions. A pH of 6 is not "normal" because it is under the upper bound --
        // it is profoundly acidaemic, and showing it as normal beside the number is worse
        // than showing nothing.
        const flag = val === undefined ? 'not measured'
          : (k.min !== null && k.min !== undefined && val < k.min) ? 'low'
          : (k.max !== null && k.max !== undefined && val > k.max) ? 'high' : 'normal';
        const typed = (st.typed.labs || {})[k.key];
        return { name: k.key,
          value: typed !== undefined ? typed : (val === undefined ? '' : String(val)),
          flag: flag,
          nameStyle: { fontWeight: 600, fontSize: '14px',
                       color: val === undefined ? '#9C99B8' : '#1B1A3A' },
          flagStyle: this.flagStyle(val === undefined ? '' : flag),
          inputStyle: { border: '1px solid #DEDCF4', borderRadius: '10px',
                        padding: '9px 12px', fontSize: '14px', background: '#FCFBFF',
                        width: '100%' },
          onChange: e => this.setNumeric('labs', k.key, e.target.value) };
      }),

      // What the module reported, once it has read an image. Before that, the row shows the
      // finding names it screens for and says nothing has been read.
      pocusFields: (st.upload ? st.upload.rows : (boot.lungFindings || []).map(n => ({
          label: n.replace(/_/g, ' '), detected: false, conf: null, caption: '' })))
        .map(r => ({
          name: r.label,
          value: r.conf === null || r.conf === undefined ? 'not read yet'
                 : (r.detected ? T('Detected ', 'Détecté ') + r.conf : T('Not detected ', 'Non détecté ') + r.conf),
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
        status: x.detected ? T('Detected', 'Détecté') : T('Not detected', 'Non détecté'),
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
        value: x.value === null ? T('Not measured', 'Non mesuré') : x.value,
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
        status: x.result === null ? T('Not measured', 'Non mesuré') : (x.flag || ''),
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
        ? T('The model backend failed and the answer was withheld. The severity and '
            + 'alerts beside it were computed before the model ran and are unaffected.',
            'Le moteur du modèle a échoué et la réponse a été retenue. La sévérité et les '
            + 'alertes à côté ont été calculées avant son exécution et ne sont pas affectées.')
        : v.differentialOrigin === 'not_generated'
        ? T('No differential was generated. Producing one requires a 4.9 GB language '
            + 'model on a GPU, which is not loaded in this deployment. Everything else on '
            + 'screen was computed here.',
            'Aucun diagnostic différentiel n’a été généré. Cela nécessite un modèle de '
            + 'langage de 4,9 Go sur GPU, non chargé dans ce déploiement. Tout le reste de '
            + 'l’écran a été calculé ici.') : '',
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
        // On the untranslated key. Matching the displayed word made every alert on a French
        // screen render as the amber "Important" variant -- including the critical ones,
        // whose red border and heading are how a clinician tells them apart at a glance.
        const crit = (a.severityKey || a.severity) === 'CRITICAL';
        return { type: a.type, message: a.message,
          kicker: crit ? T('Immediate attention', 'Attention immédiate') : T('Important', 'Important'),
          style: { background: '#fff', borderRadius: '18px', padding: '22px 26px',
            marginBottom: '16px', border: '1px solid ' + (crit ? '#F6CFCF' : '#F7DFB4'),
            borderLeft: '5px solid ' + (crit ? '#E5484D' : '#F5A623') },
          kickerStyle: { color: crit ? '#C13238' : '#9A6207', fontSize: '12.5px',
            letterSpacing: '.11em', textTransform: 'uppercase', fontWeight: 800 } };
      }),
      noAlerts: (v.alerts || []).length === 0,

      exams: (v.exams || []).map(e => ({ exam: e.exam, reason: e.reason,
                                         priority: e.priority })),
      nextStep: (v.exams || []).length
        ? T('Obtain: ', 'Obtenir : ') + v.exams[0].exam
        : T('Physician review of the current findings.',
            'Relecture médicale des données actuelles.'),
      assistantReady: T('Assistant ready', 'Assistant prêt'),
      otherLangHref: FR ? '/' : '/fr',
      otherLangLabel: FR ? '🌐 English' : '🌐 Français',
      missingChips: (v.missing || []).map(m => ({ name: m })),
      noMissing: !(v.missing || []).length,
      noMissingNote: T('Every value in the reference set was measured.',
                       'Toutes les valeurs de référence ont été mesurées.'),
      noExams: !(v.exams || []).length,
      noExamsNote: T('Nothing further is recommended: every value in the reference set was '
                     + 'measured and every expected view was obtained.',
                     'Aucun examen supplémentaire n’est recommandé : toutes les valeurs de '
                     + 'référence ont été mesurées et toutes les coupes attendues obtenues.'),
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

      // ---- the home table and the history grid list PATIENTS ------------------------
      // They used to list the project's five benchmark encounters, which are fixtures the
      // test suite runs, not people anybody assessed here. Shown under "Recent assessments"
      // beside a tile reading "0 analysed this session", they read as this doctor's patients.
      // Both screens list the encounters actually analysed in this session, and say so when
      // there are none. The benchmark still runs -- in the suite, where it belongs.
      roster: (boot.records || []).map(r => ({ name: r.name, age: String(r.age), sex: r.sex,
        complaint: r.complaint, tag: r.severity, severity: r.severity,
        alerts: r.alerts + T(' alert(s)', ' alerte(s)'),
        initials: r.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2),
        onOpen: this.openRecord(r.id),
        tagStyle: { borderRadius: '999px', padding: '5px 12px', fontSize: '12.5px',
          fontWeight: 700,
          background: r.severityKey === 'HIGH' ? '#FDECEC'
                    : r.severityKey === 'MODERATE' ? '#FFF3E0' : '#F0FADB',
          color: r.severityKey === 'HIGH' ? '#C13238'
               : r.severityKey === 'MODERATE' ? '#9A6207' : '#5A7A0F' } })),
      visits: (boot.records || []).map(r => ({ date: r.at, reason: r.name,
        meta: r.organ + ' · ' + r.alerts + T(' alert(s)', ' alerte(s)'), tag: r.severity,
        outcome: (r.findings || []).join(', ') || T('no positive finding', 'aucun signe positif'),
        onOpen: this.go('report'),
        tagStyle: { borderRadius: '999px', padding: '5px 12px', fontSize: '12.5px',
          fontWeight: 700, background: r.severityKey === 'HIGH' ? '#FDECEC' : '#F0FADB',
          color: r.severityKey === 'HIGH' ? '#C13238' : '#5A7A0F' } })),
      noRecords: !(boot.records || []).length,
      // The images the module read, filed under the patient they were read for. On the list
      // this is every patient's studies; with one open it is that patient's only — the header
      // says whose record you are in, so showing everybody's under it would misattribute them.
      gallery: (boot.records || [])
        .filter(r => !st.recordSel || r.id === st.recordSel)
        .filter(r => (r.images || []).length)
        .map(r => ({ date: r.at, reason: r.name,
          meta: r.organ + ' · ' + r.images.length + T(' stored · ', ' enregistrée(s) · ')
                + ((r.findings || []).join(', ') || T('no positive finding', 'aucun signe positif')),
          images: r.images.map(im => ({ src: im.src, finding: im.finding, zone: im.zone,
                                        time: im.time + ' · ' + im.zone })) })),
      onRecordUpload: e => this.addToRecord(e.target.files),
      noImages: !(boot.records || [])
        .filter(r => !st.recordSel || r.id === st.recordSel)
        .some(r => (r.images || []).length),
      dataRows: (v.vitals || []).concat(v.labs || []).map(x => ({
        name: x.label || x.name,
        now: x.value === null || x.result === null ? T('Not measured', 'Non mesuré')
             : String(x.value !== undefined ? x.value : x.result),
        v2: '—', v3: '—', v4: '—',
        nowStyle: { padding: '13px 0', fontWeight: 700,
          color: (x.flag && x.flag !== 'normal') ? '#C13238' : '#1B1A3A' } })),
      // Every counter reads this session. A tile counting benchmark fixtures sat beside one
      // reading "0 analysed this session", which is how five test cases came to look like
      // five patients waiting to be seen.
      sessionCount: String((boot.records || []).length),
      studyCount: String((boot.records || [])
        .reduce((n, r) => n + (r.images || []).length, 0)),
      criticalCount: String((boot.records || []).filter(r => r.severityKey === 'HIGH').length),
      testCount: boot.tests ? String(boot.tests) : '—',
      noRoster: !(boot.records || []).length,

      // ---- Diagnosis screen -----------------------------------------------------------
      // It showed "Top match / PULMONARY EDEMA / 62.4 %" beside this patient's real heart
      // rate. The reasoning agent emits a likelihood BAND and never a percentage, because a
      // number implies a calibration nothing in this pipeline has; and when no differential
      // was generated there is no top match to name at all.
      topDiagnosis: (v.differential && v.differential.length)
        ? v.differential[0].diagnosis : T('No differential', 'Aucun différentiel'),
      topLikelihood: (v.differential && v.differential.length)
        ? (v.differential[0].likelihood || 'unranked') + ' likelihood'
        : (v.withheld ? T('withheld', 'retenu') : T('not generated', 'non généré')),
      topBasis: (v.differential && v.differential.length)
        ? ((v.differential[0].supporting || []).length
           + T(' item(s) cited for it, ', ' élément(s) cité(s) en sa faveur, ')
           + (v.differential[0].contradicting || []).length
           + T(' against. ', ' en défaveur. ')
           + T('A band, not a percentage: no calibration behind this supports a number.',
               'Une fourchette, pas un pourcentage : aucune calibration ici ne justifie un '
               + 'chiffre.'))
        : (v.withheld
            ? T('The differential was withheld because the answer failed validation.',
                'Le diagnostic différentiel a été retenu : la réponse a échoué à la '
                + 'validation.')
            : T('No reasoning model ran in this deployment, so no differential was produced. '
                + 'The severity, the alerts and the escalation above were computed without it.',
                'Aucun modèle de raisonnement n’a été exécuté dans ce déploiement ; aucun '
                + 'différentiel n’a donc été produit. La sévérité, les alertes et l’escalade '
                + 'ci-dessus ont été calculées sans lui.')),
      diagnosisLine: has
        ? ((v.patient || {}).name || T('This patient', 'Ce patient')) + ' — ' + nAlerts
          + T(' alert(s), ', ' alerte(s), ') + (v.missing || []).length
          + T(' value(s) never measured', ' valeur(s) jamais mesurée(s)')
        : T('No encounter has been analysed yet.',
            'Aucune prise en charge n’a encore été analysée.'),
      examCount: String((v.exams || []).length) + T(' recommended', ' recommandé(s)'),
      anomalyCount: String((v.alerts || []).filter(
                      x => x.severity === 'CRITICAL' || x.severity === 'CRITIQUE').length)
                    + T(' critical', ' critique(s)'),

      // ---- record: reports and the measurement columns ---------------------------------
      // The tab listed three prior visits this patient never had. It lists the encounters
      // THIS session holds for them, and says so when there is only one.
      patientReports: (boot.records || [])
        .filter(r => !st.recordSel || r.name === ((v.patient || {}).name))
        .map(r => ({ at: r.at, complaint: r.complaint,
          summary: r.organ + ' · ' + r.severity + ' · ' + r.alerts + T(' alert(s)', ' alerte(s)'),
          onOpen: this.openRecord(r.id) })),
      oneReportOnly: (boot.records || [])
        .filter(r => !st.recordSel || r.name === ((v.patient || {}).name)).length < 2,
      // Dated columns for visits that never happened: a dash under "12 Jun" is a claim that
      // the date existed. There is one column, and it is this encounter.
      visitCol2: '—', visitCol3: '—', visitCol4: '—',
      timelineNote: has
        ? T('These are the stages of THIS encounter, timed as they ran. Nothing revises an '
            + 'assessment in place: analysing again produces a new encounter, so what was '
            + 'shown at the time stays comparable with what is shown now.',
            'Voici les étapes de CETTE prise en charge, chronométrées telles qu’elles se sont '
            + 'déroulées. Rien ne révise une évaluation sur place : relancer l’analyse crée '
            + 'une nouvelle prise en charge, afin que ce qui a été montré alors reste '
            + 'comparable à ce qui est montré maintenant.')
        : T('No encounter has been analysed yet, so there are no stages to show.',
            'Aucune prise en charge n’a été analysée : il n’y a aucune étape à montrer.'),
      assistantPatient: has
        ? ((v.patient || {}).age || '—') + ((v.patient || {}).sex || '')
          + ' — ' + ((v.patient || {}).complaint || 'no complaint given')
        : T('no patient', 'aucun patient'),
      ctxVitals: has ? (v.vitals || []).filter(x => x.value !== null).length
                       + T(' of ', ' sur ') + (v.vitals || []).length
                       + T(' recorded', ' relevées') : T('none', 'aucune'),
      ctxLabs: has ? (v.labs || []).filter(x => x.result !== null).length
                     + T(' of ', ' sur ') + (v.labs || []).length
                     + T(' resulted', ' rendus') : T('none', 'aucun'),
      ctxPocus: has ? ((v.patient || {}).organ || '—') + ' · '
                      + (v.findings || []).filter(f => f.detected).length
                      + T(' finding(s)', ' signe(s)')
                    : T('none', 'aucun'),
      ctxGenerated: has ? (v.generatedAt || '—') : T('not generated', 'non généré'),
      historyLine: (boot.records || []).length
        + T(' assessment(s) this session · ', ' évaluation(s) cette session · ')
        + (boot.records || []).filter(r => r.severityKey === 'HIGH').length
        + T(' high severity', ' de sévérité élevée'),
      // Printed under ONE patient's name, so they have to be that patient's. "Encounters 3 /
      // POCUS studies 2" were session totals sitting beside mariem's alerts and severity,
      // which reads as a claim about her. On the list they are the session's, where they are.
      recEncounters: String(st.recordSel
        ? (boot.records || []).filter(r => r.name === ((v.patient || {}).name)).length
        : (boot.records || []).length),
      recStudies: String(st.recordSel
        ? ((boot.records || []).filter(r => r.id === st.recordSel)[0] || {} ).images
          ? (boot.records || []).filter(r => r.id === st.recordSel)[0].images.length : 0
        : (boot.records || []).reduce((n, r) => n + (r.images || []).length, 0)),
      assistantLine: (boot.records || []).length
        ? (boot.records || []).length
          + T(' case(s) assessed in this session are open for review.',
              ' cas évalué(s) cette session sont disponibles pour relecture.')
        : T('Nothing assessed yet. The assistant answers from a computed encounter, so there '
            + 'is nothing for it to read until you analyse one.',
            'Rien n’a encore été évalué. L’assistant répond à partir d’une prise en charge '
            + 'calculée : il n’a rien à lire tant que vous n’en avez pas analysé une.'),
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
      askWhy: () => this.ask(T('Why is this the severity?', 'Pourquoi cette sévérité ?')),
      askChallenge: () => this.ask(T('Challenge this assessment.', 'Conteste cette évaluation.')),
      askBLines: () => this.ask(T('What did POCUS actually see?', 'Qu’a réellement vu l’échographie ?')),

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
      // The record screen shows the LIST until a patient is opened.
      recordList: !st.recordSel,
      recordOpen: !!st.recordSel,
      patients: (boot.records || []).map(r => ({
        name: r.name, age: String(r.age), sex: r.sex, complaint: r.complaint, at: r.at,
        severity: r.severity, alerts: r.alerts + T(' alert(s)', ' alerte(s)'), organ: r.organ,
        encounterId: r.encounterId,
        findings: ((r.findings || []).join(', ') || T('no positive finding', 'aucun signe positif'))
                  + ((r.images || []).length ? ' · ' + r.images.length
                     + T(' image(s)', ' image(s)') : ''),
        initials: r.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2),
        onOpen: this.openRecord(r.id),
        tagStyle: { borderRadius: '999px', padding: '5px 12px', fontSize: '12.5px',
          fontWeight: 700,
          background: r.severityKey === 'HIGH' ? '#FDECEC'
                    : r.severityKey === 'MODERATE' ? '#FFF3E0' : '#F0FADB',
          color: r.severityKey === 'HIGH' ? '#C13238'
               : r.severityKey === 'MODERATE' ? '#9A6207' : '#5A7A0F' } })),
      noPatients: !(boot.records || []).length,
      patientCount: String((boot.records || []).length),
      backToList: () => this.backToList(),

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
