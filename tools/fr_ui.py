"""French for the interface chrome.

Only the interface is here: headings, labels, buttons, empty states and the explanatory copy
the screens carry. Every clinical sentence -- alerts, escalation triggers, recommended
examinations, declared limits, the conclusion -- is rendered by src/agents/i18n.py from the
same computed fields the English was built from, and never by translating an English sentence.
The split matters: chrome is written by hand and can safely be phrased freely, while a clinical
statement has to be composed from the numbers it came from or it is a new claim.

Applied by tools/bind_design.py, which produces a second page rather than translating at
runtime. A page per language is checked by the same balance and fabrication checks as the
English one, and costs nothing to serve.

`assert_covered` fails the build if a visible English string has no entry, so a new sentence
added to the design cannot reach a French screen in English by being forgotten. That is the
same property the clinical layer's coverage test enforces, for the same reason.
"""
from __future__ import annotations

UI: dict[str, str] = {
    # ---- shell -----------------------------------------------------------------------
    "POCUS-Emergency": "POCUS-Emergency",
    "Clinical Copilot": "Copilote clinique",
    "Current patient": "Patient en cours",
    "Perception modules": "Modules de perception",
    "Search patients, findings, reports…": "Rechercher patients, signes, rapports…",
    "Assistant ready": "Assistant prêt",
    "+ New assessment": "+ Nouvelle évaluation",

    # ---- home ------------------------------------------------------------------------
    "Your clinical copilot": "Votre copilote clinique",
    "Start a new clinical assessment using clinical information, vital signs, laboratory "
    "results and POCUS findings. The assistant highlights what needs attention and what "
    "information is missing.":
        "Démarrez une évaluation clinique à partir des informations cliniques, des constantes, "
        "des résultats biologiques et de l’échographie. L’assistant met en évidence ce qui "
        "demande attention et ce qui manque.",
    "+ New patient assessment": "+ Évaluer un nouveau patient",
    "Ask the assistant": "Interroger l’assistant",
    "Patients this session": "Patients cette session",
    "POCUS studies read": "Examens POCUS lus",
    "High priority": "Priorité haute",
    "Safety tests passing": "Tests de sécurité réussis",
    "Start an assessment": "Démarrer une évaluation",
    "Enter a new patient and analyze the case.":
        "Saisir un nouveau patient et analyser le cas.",
    "Start →": "Démarrer →",
    "Review a case": "Revoir un cas",
    "Continue the current encounter.": "Poursuivre la prise en charge en cours.",
    "Open case →": "Ouvrir le cas →",
    "Discuss the case in plain language.": "Discuter du cas en langage courant.",
    "Open assistant →": "Ouvrir l’assistant →",
    "Generate a report": "Générer un rapport",
    "Create a standardized clinical summary.": "Créer un compte rendu clinique standardisé.",
    "Reports →": "Rapports →",
    "Recent assessments": "Évaluations de cette session",
    "View all →": "Tout afficher →",
    "Patient": "Patient",
    "Age": "Âge",
    "Reason": "Motif",
    "Status": "Statut",
    "Time": "Heure",
    "No patient has been assessed in this session. Open Patient workup to enter one.":
        "Aucun patient n’a été évalué durant cette session. Ouvrez « Saisie du patient » "
        "pour en saisir un.",
    "Clinical assistant": "Assistant clinique",
    "Review cases →": "Revoir les cas →",

    # ---- workup ----------------------------------------------------------------------
    "Patient workup": "Saisie du patient",
    "Enter everything you have on one page — history, vital signs, laboratory results and "
    "POCUS. Then analyze. Anything left blank is recorded as not provided, never as normal.":
        "Saisissez tout ce dont vous disposez sur une seule page — anamnèse, constantes, "
        "biologie et échographie — puis analysez. Tout champ laissé vide est enregistré comme "
        "non renseigné, jamais comme normal.",
    "1 · Patient &amp; history": "1 · Patient et anamnèse",
    "Name": "Nom",
    "Sex": "Sexe",
    "Female": "Femme",
    "Male": "Homme",
    "Chief complaint": "Motif de consultation",
    "Dyspnea": "Dyspnée",
    "Fatigue": "Asthénie",
    "+ Fever": "+ Fièvre",
    "+ Chest pain": "+ Douleur thoracique",
    "+ Cough": "+ Toux",
    "Medical history &amp; medications": "Antécédents et traitements",
    "2 · Vital signs": "2 · Constantes",
    "3 · Laboratory": "3 · Biologie",
    "+ Add test": "+ Ajouter un examen",
    "4 · POCUS": "4 · Échographie (POCUS)",
    "5 · Analyze": "5 · Analyser",
    "The assistant reads everything on this page at once and returns a ranked assessment with "
    "what supports it, what limits confidence, and what is missing.":
        "L’assistant lit l’ensemble de cette page en une fois et renvoie une évaluation "
        "hiérarchisée : ce qui l’étaye, ce qui limite la confiance, et ce qui manque.",
    "Analyzing this patient…": "Analyse de ce patient…",
    "Review alerts →": "Voir les alertes →",

    # ---- assessment ------------------------------------------------------------------
    "Assessment": "Évaluation",
    "Offered for consideration. This is not a diagnosis.":
        "Proposé à titre indicatif. Ceci n’est pas un diagnostic.",
    "Generate report": "Générer le rapport",
    "Supports": "En faveur",
    "Limits confidence": "Limite la confiance",
    "What would clarify this case": "Ce qui clarifierait ce cas",
    "Not measured does not mean normal.": "Non mesuré ne veut pas dire normal.",
    "Suggested next step": "Étape suivante suggérée",
    "Ask the assistant →": "Interroger l’assistant →",
    "Clinical assessment": "Évaluation clinique",
    "Generate clinical report": "Générer le compte rendu",
    "What the assistant sees": "Ce que l’assistant voit",
    "The information available for this patient.":
        "Les informations disponibles pour ce patient.",
    "Symptoms": "Symptômes",
    "Vital signs": "Constantes",
    "POCUS": "POCUS",
    "Laboratory": "Biologie",
    "Differential assessment": "Évaluation différentielle",
    "Why it is considered": "Pourquoi cette hypothèse",
    "What limits confidence": "Ce qui limite la confiance",
    "Why?": "Pourquoi ?",
    "Challenge this assessment": "Contester cette évaluation",
    "What would help clarify this case?": "Qu’est-ce qui clarifierait ce cas ?",
    "These tests were not performed. Their absence does not mean they are normal.":
        "Ces examens n’ont pas été réalisés. Leur absence ne signifie pas qu’ils sont normaux.",
    "✨ Ask the clinical assistant": "✨ Interroger l’assistant clinique",
    "Discuss this case in plain language — what supports the leading possibility, what argues "
    "against it, what is still missing.":
        "Discutez du cas en langage courant : ce qui soutient l’hypothèse principale, ce qui "
        "s’y oppose, et ce qui manque encore.",
    "Open assistant": "Ouvrir l’assistant",

    # ---- diagnosis -------------------------------------------------------------------
    "▤ Evidence cited": "▤ Éléments de preuve cités",
    "alerts raised": "alertes déclenchées",
    "▦ Differential diagnosis — supporting vs against":
        "▦ Diagnostic différentiel — en faveur / en défaveur",
    "Supporting": "En faveur",
    "Against": "En défaveur",
    "ALL": "TOUT",
    "Bars show how strongly the recorded findings support or argue against each possibility. "
    "Ranges stay wide where investigations are unavailable.":
        "Les barres indiquent la force avec laquelle les éléments recueillis soutiennent ou "
        "contredisent chaque hypothèse. Les fourchettes restent larges lorsque des examens "
        "manquent.",
    "⌁ Top match": "⌁ Hypothèse principale",
    "✛ Heart rate": "✛ Fréquence cardiaque",
    "bpm": "bpm",
    "〜 Oxygen saturation": "〜 Saturation en oxygène",
    "Add measurement": "Ajouter une mesure",

    # ---- record ----------------------------------------------------------------------
    "Patients": "Patients",
    "No patient has been assessed yet. Open Patient workup, enter what you have and analyse "
    "the case.":
        "Aucun patient n’a encore été évalué. Ouvrez « Saisie du patient », renseignez ce dont "
        "vous disposez et analysez le cas.",
    "← All patients": "← Tous les patients",
    "Encounters": "Prises en charge",
    "POCUS studies": "Examens POCUS",
    "Alerts": "Alertes",
    "Severity": "Sévérité",
    "Stored POCUS images": "Images POCUS enregistrées",
    "session-scoped · closing the app discards it":
        "limité à la session · la fermeture de l’application efface tout",
    "No study has been read for this patient. Uploading one on Patient workup files it here "
    "with what the module made of it.":
        "Aucun examen n’a été lu pour ce patient. En téléverser un depuis « Saisie du "
        "patient » le classe ici avec ce que le module en a conclu.",
    "＋ Add images to this patient": "＋ Ajouter des images à ce patient",
    "Read by the module now and filed against this patient, for this session only — there is "
    "no database behind this screen. The assessment above is not re-run: it was reached "
    "without this study.":
        "Lu par le module maintenant et classé au dossier de ce patient, pour cette session "
        "uniquement — il n’y a pas de base de données derrière cet écran. L’évaluation "
        "ci-dessus n’est pas relancée : elle a été établie sans cet examen.",
    "Stored measurements — all visits": "Mesures enregistrées",
    "Measurement": "Mesure",
    "Today": "Aujourd’hui",
    "Blank cells mean the measurement was not taken at that visit. Not measured does not mean "
    "normal.":
        "Une case vide signifie que la mesure n’a pas été prise. Non mesuré ne veut pas dire "
        "normal.",
    "Visit history": "Historique des prises en charge",
    "Open report →": "Ouvrir le rapport →",
    "Stored reports": "Rapports enregistrés",
    "Open →": "Ouvrir →",
    "This session holds no earlier encounter for this patient. There is no database behind "
    "this screen: it shows what was assessed here, not a medical history.":
        "Cette session ne contient aucune prise en charge antérieure pour ce patient. Il n’y "
        "a pas de base de données derrière cet écran : il montre ce qui a été évalué ici, pas "
        "un dossier médical.",

    # ---- alerts ----------------------------------------------------------------------
    "Clinical alerts": "Alertes cliniques",
    "Acknowledge non-critical": "Acquitter les non critiques",
    "Review patient": "Revoir le patient",
    "No alert": "Aucune alerte",
    "No measured value crossed a configured bound and the record shows no structural gap.":
        "Aucune valeur mesurée n’a franchi une borne configurée et le dossier ne présente "
        "aucune lacune structurelle.",
    "Escalation triggers": "Déclencheurs d’escalade",
    "Evaluated before the model runs": "Évalués avant l’exécution du modèle",
    "Information missing": "Informations manquantes",
    "Important investigations have not been performed":
        "Des examens importants n’ont pas été réalisés",
    "These tests have not been performed. They should not be interpreted as normal.":
        "Ces examens n’ont pas été réalisés. Ils ne doivent pas être interprétés comme normaux.",
    "View missing information →": "Voir les informations manquantes →",

    # ---- assistant -------------------------------------------------------------------
    "I'm here to help you review this case.":
        "Je suis là pour vous aider à revoir ce cas.",
    "Patient context": "Contexte du patient",
    "The assistant keeps everything entered during this encounter, so you don't need to "
    "repeat the patient information.":
        "L’assistant conserve tout ce qui a été saisi pour cette prise en charge ; vous n’avez "
        "pas à répéter les informations du patient.",
    "Vitals": "Constantes",
    "Reminder": "Rappel",
    "The assistant provides clinical decision support. It does not make a diagnosis, and "
    "responsibility for the decision rests with you.":
        "L’assistant fournit une aide à la décision clinique. Il ne pose pas de diagnostic, et "
        "la responsabilité de la décision vous revient.",

    # ---- timeline --------------------------------------------------------------------
    "Patient clinical timeline": "Chronologie clinique du patient",
    "One encounter, computed once": "Une prise en charge, calculée une fois",

    # ---- report ----------------------------------------------------------------------
    "Clinical assessment report": "Compte rendu d’évaluation clinique",
    "🖨 Print": "🖨 Imprimer",
    "📄 Export PDF": "📄 Exporter en PDF",
    "💾 Save to record": "💾 Enregistrer au dossier",
    "Clinical summary": "Synthèse clinique",
    "Clinical findings": "Données cliniques",
    "POCUS &amp; laboratory": "POCUS et biologie",
    "Differential": "Diagnostic différentiel",
    "Important limitations": "Limites importantes",
    "Not measured does not mean normal": "Non mesuré ne veut pas dire normal",
    "Recommended next step": "Étape suivante recommandée",
    "AI-assisted clinical assessment. Requires physician review. This report is clinical "
    "decision support and does not constitute a diagnosis.":
        "Évaluation clinique assistée par IA. Nécessite une relecture médicale. Ce compte "
        "rendu est une aide à la décision et ne constitue pas un diagnostic.",

    # ---- history ---------------------------------------------------------------------
    "My clinical assessments": "Mes évaluations cliniques",
    "All": "Toutes",
    "Critical": "Critiques",
    "Review": "À revoir",
    "Completed": "Terminées",
    "⌕ Search patient…": "⌕ Rechercher un patient…",
    "Nothing has been assessed in this session yet. There is no database behind this screen: "
    "closing the app discards it.":
        "Rien n’a encore été évalué durant cette session. Il n’y a pas de base de données "
        "derrière cet écran : la fermeture de l’application efface tout.",
}

# Strings that are the same in both languages, or are not language at all. Listed explicitly
# so that `assert_covered` stays a real check rather than one with a silent escape hatch.
SAME = {
    "POCUS-Emergency", "POCUS", "bpm", "ALL", "Patient", "Name", "Age", "Alerts", "Vitals",
    "Differential", "Measurement", "Today", "All", "Critical", "Review", "Completed",
    "Assessment", "Severity", "Encounters", "Status", "Time", "Reason", "Sex", "Fatigue",
    "Supporting", "Against", "Supports",
}


def translate(html: str) -> str:
    """Replace every known English string in the page with its French form.

    Longest first: "Alerts" is a substring of longer sentences containing it, and replacing
    the short one first would corrupt the long one.
    """
    out = html
    for en in sorted(UI, key=len, reverse=True):
        fr = UI[en]
        if fr != en:
            out = out.replace(f">{en}<", f">{fr}<")
            out = out.replace(f'placeholder="{en}"', f'placeholder="{fr}"')
    return out


def assert_covered(visible: list[str]) -> list[str]:
    """Every visible English string must have an entry. Returns what does not."""
    return [t for t in visible if t not in UI and t not in SAME]
