import os

import numpy as np

from openmdao.api import Problem, write_xdsm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from openmdao.test_suite.components.sellar import SellarNoDerivatives

try:
    from pyxdsm.XDSM import XDSM
except ImportError:
    XDSM = None

"""Makes XDSM for the Sellar problem"""
from openmdao.recorders.sqlite_recorder import SqliteRecorder

filename = 'xdsm_from_sql'
case_recording_filename = filename + '.sql'

prob = Problem()
prob.model = model = SellarNoDerivatives()
model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                     upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
model.add_design_var('x', lower=0.0, upper=10.0)
model.add_objective('obj')
model.add_constraint('con1', equals=np.zeros(1))
model.add_constraint('con2', upper=0.0)
prob.driver = ScipyOptimizeDriver()

recorder = SqliteRecorder(case_recording_filename)
prob.driver.add_recorder(recorder)

prob.setup(check=False)
prob.run_model()

# Write output
write_xdsm(case_recording_filename, filename=filename, out_format='pdf', show_browser=True,
           quiet=False)

# Check if file was created
assert os.path.isfile('.'.join([filename, 'tex']))