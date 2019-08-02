import os

import numpy as np

import openmdao.api as om
from openmdao.api import Problem, write_xdsm
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.test_suite.components.sellar_feature import SellarMDA

filename = '_xdsm_sellar'
prob = Problem(model=SellarMDA())
model = prob.model
model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                     upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
model.add_design_var('x', lower=0.0, upper=10.0)
model.add_objective('obj')
model.add_constraint('con1', equals=np.zeros(1))
model.add_constraint('con2', upper=0.0)
model.nonlinear_solver = NewtonSolver()
syss = list(model.system_iter(include_self=False, recurse=True))
# cycle = syss[0]
# model.cycle.nonlinear_solver = NonlinearBlockGS()

prob.setup(check=False)
prob.final_setup()

# Write output
om.n2(prob)
# write_xdsm(prob, filename=filename, out_format='pdf', show_browser=True, quiet=False,
#            include_solver=True, add_process_conns=True,
#            box_stacking='horizontal', number_alignment="vertical")
# write_xdsm(prob, filename=filename+'2', out_format='pdf', show_browser=True, quiet=False,
#            include_solver=True, add_process_conns=True,
#            box_stacking='vertical', number_alignment="horizontal")
# write_xdsm(prob, filename=filename+'3', out_format='pdf', show_browser=True, quiet=False,
#            include_solver=True, add_process_conns=True,
#            box_stacking='vertical', numbered_comps=False)
# write_xdsm(prob, filename=filename+'4', out_format='pdf', show_browser=True, quiet=False,
#            include_solver=True, add_process_conns=True,
#            box_stacking='vertical', numbered_comps=False, class_names=False)

write_xdsm(prob, filename=filename, out_format='html', show_browser=True, quiet=False,
           include_solver=True, add_process_conns=True,
           box_stacking='horizontal', number_alignment="vertical")

# write_xdsm(prob, filename=filename, out_format='html', show_browser=True,
#            include_solver=True)
# write_xdsm(prob, filename=filename+'2', out_format='pdf', show_browser=True,
#            include_solver=True, recurse=False)
# my_writer = XDSMjsWriter(name='may_xdsmjs_writer')
# write_xdsm(prob, filename=filename, writer=my_writer, show_browser=True, quiet=False,
#            include_solver=True)

# Check if file was created
assert os.path.isfile('.'.join([filename, 'tex']))