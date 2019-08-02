import os

from openmdao.api import Problem, IndepVarComp
from openmdao.api import Group
from openmdao.devtools.xdsm_viewer.xdsm_writer import write_xdsm
from openmdao.test_suite.scripts.circuit import Circuit

p = Problem()
model = p.model

group = model.add_subsystem('G1', Group(), promotes=['*'])
group2 = model.add_subsystem('G2', Group())
group.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
group.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
group2.add_subsystem('source2', IndepVarComp('I', 0.1, units='A'))
group.add_subsystem('circuit', Circuit())

group.connect('source.I', 'circuit.I_in')
group.connect('ground.V', 'circuit.Vg')

# model.add_design_var('ground.V')
# model.add_design_var('source.I')
# model.add_objective('circuit.D1.I')

p.setup(check=False)

# set some initial guesses
p['circuit.n1.V'] = 10.
p['circuit.n2.V'] = 1.

p.run_model()

write_xdsm(p, 'xdsm_circuit2', out_format='pdf', quiet=True, show_browser=False,
           recurse=True, model_path='G1')
assert os.path.isfile('.'.join(['xdsm_circuit2', 'tex']))
