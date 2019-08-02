import numpy as np
from openmdao.components.exec_comp import ExecComp

from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ArmijoGoldsteinLS
from openmdao.test_suite.components.sellar_feature import SellarDis1, SellarDis2


p = Problem()
model = p.model
indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
indeps.add_output('x', 1.0)
indeps.add_output('z', np.array([5.0, 2.0]))

cycle = model.add_subsystem('cycle', Group(), promotes=['*'])
cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'], promotes_outputs=['y1'])
cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'], promotes_outputs=['y2'])

# Nonlinear Block Gauss Seidel is a gradient free solver
cycle.nonlinear_solver = NewtonSolver()

model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                       z=np.array([0.0, 0.0]), x=0.0),
                   promotes=['x', 'z', 'y1', 'y2', 'obj'])

model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])
p.model.cycle.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
p.setup()
p.run_model()