"""
Example simple problem
"""

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, ExecComp, ScipyOptimizeDriver

try:
    from pyxdsm.XDSM import XDSM
    from openmdao.devtools.xdsm_viewer.xdsm_writer import write_xdsm, write_html
except ImportError:
    XDSM = None

FILENAME = 'opt_example'

class Rosenbrock(ExplicitComponent):

    def __init__(self, problem):
        super(Rosenbrock, self).__init__()
        self.problem = problem
        self.counter = 0

    def setup(self):
        self.add_input('x', np.array([1.5, 1.5]))
        self.add_output('f', 0.0)
        self.declare_partials('f', 'x', method='fd', form='central', step=1e-4)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs['x']
        outputs['f'] = sum(x ** 2)


x0 = np.array([1.2, 1.5])
filename = FILENAME + '2'

prob = Problem()
indeps = prob.model.add_subsystem('indeps', IndepVarComp(problem=prob), promotes=['*'])
indeps.add_output('x', list(x0))

prob.model.add_subsystem('sphere', Rosenbrock(problem=prob), promotes=['*'])
prob.model.add_subsystem('con', ExecComp('c=sum(x)', x=np.ones(2)), promotes=['*'])
prob.driver = ScipyOptimizeDriver()
prob.model.add_design_var('x')
prob.model.add_objective('f')
prob.model.add_constraint('c', lower=1.0)

prob.setup(check=False)
prob.final_setup()

# Write output
write_xdsm(prob, filename=filename, out_format='tex', show_browser=True)