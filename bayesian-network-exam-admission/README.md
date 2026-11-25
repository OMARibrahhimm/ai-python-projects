# Bayesian Network: Exam Admission

A small Bayesian network built with [pyAgrum](https://pyagrum.readthedocs.io/) that models how exam level and IQ level affect marks, aptitude score, and admission.

```
examLevel -> marks -> admission
iqLevel   -> marks
iqLevel   -> aptitude
```

The script defines the conditional probability tables, runs inference with `LazyPropagation`, and prints the probabilities, including an example query with evidence (`iqLevel = 1`).

```bash
pip install pyagrum
python bayesian_network.py
```
