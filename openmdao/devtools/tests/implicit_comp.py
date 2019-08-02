from openmdao.api import Problem, IndepVarComp
from openmdao.devtools.xdsm_viewer.xdsm_writer import write_xdsm
from openmdao.test_suite.scripts.circuit import Circuit

p = Problem()
model = p.model

model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
model.add_subsystem('circuit', Circuit())

model.connect('source.I', 'circuit.I_in')
model.connect('ground.V', 'circuit.Vg')

model.add_design_var('ground.V')
model.add_design_var('source.I')
model.add_objective('circuit.D1.I')

p.setup(check=False)

# set some initial guesses
p['circuit.n1.V'] = 10.
p['circuit.n2.V'] = 1.

p.run_model()

write_xdsm(p, 'xdsm_circuit_implicit', out_format='pdf', show_browser=True,
           recurse=True, legend=True)
