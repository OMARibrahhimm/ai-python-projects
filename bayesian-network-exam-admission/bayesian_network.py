import pyagrum as gum
import numpy as np


bn = gum.BayesNet('ExamAdmission')
student = bn.add(gum.LabelizedVariable('student', 'Student', 2))
examLevel = bn.add(gum.LabelizedVariable('examLevel', 'Exam Level', 2))
iqLevel = bn.add(gum.LabelizedVariable('iqLevel', 'IQ Level', 2))
marks = bn.add(gum.LabelizedVariable('marks', 'Marks', 2))
aptitude = bn.add(gum.LabelizedVariable('aptitude', 'Aptitude Score', 2))
admission = bn.add(gum.LabelizedVariable('admission', 'Admission', 2))
bn.addArc(examLevel, marks)
bn.addArc(iqLevel, marks)
bn.addArc(iqLevel, aptitude)
bn.addArc(marks, admission)

bn.cpt(examLevel).fillWith([0.7, 0.3])
bn.cpt(iqLevel).fillWith([0.8, 0.2])

bn.cpt(aptitude)[{'iqLevel': 0}] = [0.75, 0.25]
bn.cpt(aptitude)[{'iqLevel': 1}] = [0.40, 0.60]

bn.cpt(marks)[{'iqLevel': 0, 'examLevel': 0}] = [0.6, 0.4]
bn.cpt(marks)[{'iqLevel': 0, 'examLevel': 1}] = [0.9, 0.1]
bn.cpt(marks)[{'iqLevel': 1, 'examLevel': 0}] = [0.5, 0.5]
bn.cpt(marks)[{'iqLevel': 1, 'examLevel': 1}] = [0.8, 0.2]

bn.cpt(admission)[{'marks': 0}] = [0.6, 0.4]
bn.cpt(admission)[{'marks': 1}] = [0.9, 0.1]

model = gum.LazyPropagation(bn)
model.makeInference()


# Prior probability of each variable after inference
for name in ("marks", "aptitude", "admission"):
    print(name, model.posterior(name).tolist())

# Example query: how does admission change for a student with high IQ?
model.setEvidence({"iqLevel": 1})
model.makeInference()
print("admission | iqLevel=1 ->", model.posterior("admission").tolist())
