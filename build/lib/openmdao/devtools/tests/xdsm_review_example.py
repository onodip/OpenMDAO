"""
XDSM writer review

Comments by Kenneth T Moore:

There are just a couple of things I noticed.

If you have an underscore in a variable or component name, the LaTeX, it is replaced by a space in
most blocks, which looks kind of strange. I would prefer underscores there.  OK
In the output blocks
(white parallelograms), they aren't escaped properly, so the character after the underscore becomes
a subscript.

If you have a lot of long variable names in the intermediate connections, the blocks become very
long. I like how the variables are stacked vertically in the connections that come from the green
Driver block, so it might be nice to stack inter-component connections that way too, once they are
longer than a certain length.

Found some missing connections in a dymos model. Dymos does things a little differently than a
conventional model, but I can see the connections when I run view_model, so they are real. I am
going to try to reduce this to something that doesn't contain Dymos to see if I can isolate the bug.

What I think is happening is the xdsm writer is getting confused when there are multiple
instances of a group, so that there are components and connections with the same local names
(though they have unique global names). The connection info is all being packed into the last
group's connection. Compare the output of the model viewer with the output of the xdsm writer in
the example above, and you'll see. Probably an easy fix.
"""

from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent
from openmdao.api import view_model
from openmdao.devtools.xdsm_viewer.xdsm_writer import write_xdsm
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver


class TimeComp(ExplicitComponent):

    def setup(self):
        self.add_input('t_initial', val=0.)
        self.add_input('t_duration', val=1.)
        self.add_output('time', shape=(2, ))

    def compute(self, inputs, outputs):
        t_initial = inputs['t_initial']
        t_duration = inputs['t_duration']

        outputs['time'][0] = t_initial
        outputs['time'][1] = t_initial + t_duration


class Phase(Group):

    def setup(self):
        super(Phase, self).setup()

        indep = IndepVarComp()
        for var in ['t_initial', 't_duration']:
            indep.add_output(var, val=1.0)

        self.add_subsystem('time_extents', indep, promotes_outputs=['*'])

        time_comp = TimeComp()
        self.add_subsystem('time', time_comp)

        self.connect('t_initial', 'time.t_initial')
        self.connect('t_duration', 'time.t_duration')

        self.set_order(['time_extents', 'time'])


p = Problem()
p.driver = ScipyOptimizeDriver()
orbit_phase = Phase()
p.model.add_subsystem('orbit_phase', orbit_phase)

systems_phase = Phase()
p.model.add_subsystem('systems_phase', systems_phase)

systems_phase = Phase()
p.model.add_subsystem('extra_phase', systems_phase)
p.model.add_design_var('orbit_phase.t_initial')
p.model.add_design_var('orbit_phase.t_duration')
p.setup(check=True)

p.run_model()

view_model(p)

# add_process = True
# write_xdsm(p, 'zzz', out_format='tex', box_stacking='vertical', box_width=15, box_lines=10,
#            add_process_conns=add_process)
# write_xdsm(p, 'zzz', out_format='html', add_process_conns=add_process)
print('done')