import os

from openmdao.core.problem import Problem
from openmdao.devtools.xdsm_viewer.xdsm_writer import write_xdsm
from openmdao.test_suite.components.sellar import SellarNoDerivatives

import numpy as np

QUIET = True

filename = 'xdsm_outputs_on_the_right'
prob = Problem()
prob.model = model = SellarNoDerivatives()
model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                     upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
model.add_design_var('x', lower=0.0, upper=10.0)
model.add_objective('obj')
model.add_constraint('con1', equals=np.zeros(1))
model.add_constraint('con2', upper=0.0)

prob.setup(check=False)
prob.final_setup()

# Write output
write_xdsm(prob, filename=filename, out_format='pdf', show_browser=False, quiet=QUIET,
           output_side='right')

# Check if file was created
assert os.path.isfile('.'.join([filename, 'tex']))

filename = 'xdsm_outputs_side_mixed'
# Write output
write_xdsm(prob, filename=filename, out_format='pdf', show_browser=False, quiet=QUIET,
           output_side={'optimization': 'left', 'default': 'right'})

# Check if file was created
assert os.path.isfile('.'.join([filename, 'tex']))
