"""Doctor accounts, patients and stored examinations.

The clinical pipeline under src/agents/ does not import anything from here and does not know
that accounts exist. This package answers "whose data is this?"; that one answers "what does
the evidence support?", and keeping the two apart is why adding authentication changed no
model, threshold or reported result.
"""
