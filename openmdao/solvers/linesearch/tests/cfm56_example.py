from pycycle.example_cycles import CFM56

import time
from openmdao.api import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.utils.units import convert_units as cu

prob = Problem()

des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

# FOR DESIGN
des_vars.add_output('alt', 35000., units='ft'),
des_vars.add_output('MN', 0.8),
des_vars.add_output('T4max', 2857.0, units='degR'),
des_vars.add_output('Fn_des', 5500.0, units='lbf'),

des_vars.add_output('inlet:ram_recovery', 0.9990),
des_vars.add_output('inlet:MN_out', 0.751),
des_vars.add_output('fan:PRdes', 1.685),
des_vars.add_output('fan:effDes', 0.8948),
des_vars.add_output('fan:MN_out', 0.4578)
des_vars.add_output('splitter:BPR', 5.105),
des_vars.add_output('splitter:MN_out1', 0.3104)
des_vars.add_output('splitter:MN_out2', 0.4518)
des_vars.add_output('duct4:dPqP', 0.0048),
des_vars.add_output('duct4:MN_out', 0.3121),
des_vars.add_output('lpc:PRdes', 1.935),
des_vars.add_output('lpc:effDes', 0.9243),
des_vars.add_output('lpc:MN_out', 0.3059),
des_vars.add_output('duct6:dPqP', 0.0101),
des_vars.add_output('duct6:MN_out', 0.3563),
des_vars.add_output('hpc:PRdes', 9.369),
des_vars.add_output('hpc:effDes', 0.8707),
des_vars.add_output('hpc:MN_out', 0.2442),
des_vars.add_output('bld3:MN_out', 0.3000)
des_vars.add_output('burner:dPqP', 0.0540),
des_vars.add_output('burner:MN_out', 0.1025),
des_vars.add_output('hpt:effDes', 0.8888),
des_vars.add_output('hpt:MN_out', 0.3650),
des_vars.add_output('duct11:dPqP', 0.0051),
des_vars.add_output('duct11:MN_out', 0.3063),
des_vars.add_output('lpt:effDes', 0.8996),
des_vars.add_output('lpt:MN_out', 0.4127),
des_vars.add_output('duct13:dPqP', 0.0107),
des_vars.add_output('duct13:MN_out', 0.4463),
des_vars.add_output('core_nozz:Cv', 0.9933),
des_vars.add_output('bypBld:frac_W', 0.005),
des_vars.add_output('bypBld:MN_out', 0.4489),
des_vars.add_output('duct15:dPqP', 0.0149),
des_vars.add_output('duct15:MN_out', 0.4589),
des_vars.add_output('byp_nozz:Cv', 0.9939),
des_vars.add_output('lp_shaft:Nmech', 4666.1, units='rpm'),
des_vars.add_output('hp_shaft:Nmech', 14705.7, units='rpm'),
des_vars.add_output('hp_shaft:HPX', 250.0, units='hp'),

des_vars.add_output('hpc:cool1:frac_W', 0.050708),
des_vars.add_output('hpc:cool1:frac_P', 0.5),
des_vars.add_output('hpc:cool1:frac_work', 0.5),
des_vars.add_output('hpc:cool2:frac_W', 0.020274),
des_vars.add_output('hpc:cool2:frac_P', 0.55),
des_vars.add_output('hpc:cool2:frac_work', 0.5),
des_vars.add_output('bld3:cool3:frac_W', 0.067214),
des_vars.add_output('bld3:cool4:frac_W', 0.101256),
des_vars.add_output('hpc:cust:frac_W', 0.0445),
des_vars.add_output('hpc:cust:frac_P', 0.5),
des_vars.add_output('hpc:cust:frac_work', 0.5),
des_vars.add_output('hpt:cool3:frac_P', 1.0),
des_vars.add_output('hpt:cool4:frac_P', 0.0),
des_vars.add_output('lpt:cool1:frac_P', 1.0),
des_vars.add_output('lpt:cool2:frac_P', 0.0),

# OFF DESIGN
des_vars.add_output('OD_MN', [0.8, 0.8, 0.25, 0.00001]),
des_vars.add_output('OD_alt', [35000.0, 35000.0, 0.0, 0.0], units='ft'),
des_vars.add_output('OD_Fn_target', [5500.0, 5970.0, 22590.0, 27113.0], units='lbf'),  # 8950.0
des_vars.add_output('OD_dTs', [0.0, 0.0, 27.0, 27.0], units='degR')
des_vars.add_output('OD_cust_fracW', [0.0445, 0.0422, 0.0177, 0.0185])

# DESIGN CASE
prob.model.add_subsystem('DESIGN', CFM56(statics=True))

prob.model.connect('alt', 'DESIGN.fc.alt')
prob.model.connect('MN', 'DESIGN.fc.MN')
prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')

prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
prob.model.connect('fan:PRdes', 'DESIGN.fan.PR')
prob.model.connect('fan:effDes', 'DESIGN.fan.eff')
prob.model.connect('fan:MN_out', 'DESIGN.fan.MN')
prob.model.connect('splitter:BPR', 'DESIGN.splitter.BPR')
prob.model.connect('splitter:MN_out1', 'DESIGN.splitter.MN1')
prob.model.connect('splitter:MN_out2', 'DESIGN.splitter.MN2')
prob.model.connect('duct4:dPqP', 'DESIGN.duct4.dPqP')
prob.model.connect('duct4:MN_out', 'DESIGN.duct4.MN')
prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')
prob.model.connect('duct6:dPqP', 'DESIGN.duct6.dPqP')
prob.model.connect('duct6:MN_out', 'DESIGN.duct6.MN')
prob.model.connect('hpc:PRdes', 'DESIGN.hpc.PR')
prob.model.connect('hpc:effDes', 'DESIGN.hpc.eff')
prob.model.connect('hpc:MN_out', 'DESIGN.hpc.MN')
prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')
prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')
prob.model.connect('duct11:dPqP', 'DESIGN.duct11.dPqP')
prob.model.connect('duct11:MN_out', 'DESIGN.duct11.MN')
prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')
prob.model.connect('duct13:dPqP', 'DESIGN.duct13.dPqP')
prob.model.connect('duct13:MN_out', 'DESIGN.duct13.MN')
prob.model.connect('core_nozz:Cv', 'DESIGN.core_nozz.Cv')
prob.model.connect('bypBld:MN_out', 'DESIGN.byp_bld.MN')
prob.model.connect('duct15:dPqP', 'DESIGN.duct15.dPqP')
prob.model.connect('duct15:MN_out', 'DESIGN.duct15.MN')
prob.model.connect('byp_nozz:Cv', 'DESIGN.byp_nozz.Cv')
prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')
prob.model.connect('hp_shaft:HPX', 'DESIGN.hp_shaft.HPX')

prob.model.connect('hpc:cool1:frac_W', 'DESIGN.hpc.cool1:frac_W')
prob.model.connect('hpc:cool1:frac_P', 'DESIGN.hpc.cool1:frac_P')
prob.model.connect('hpc:cool1:frac_work', 'DESIGN.hpc.cool1:frac_work')
prob.model.connect('hpc:cool2:frac_W', 'DESIGN.hpc.cool2:frac_W')
prob.model.connect('hpc:cool2:frac_P', 'DESIGN.hpc.cool2:frac_P')
prob.model.connect('hpc:cool2:frac_work', 'DESIGN.hpc.cool2:frac_work')
prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')
prob.model.connect('bld3:cool4:frac_W', 'DESIGN.bld3.cool4:frac_W')
prob.model.connect('hpc:cust:frac_W', 'DESIGN.hpc.cust:frac_W')
prob.model.connect('hpc:cust:frac_P', 'DESIGN.hpc.cust:frac_P')
prob.model.connect('hpc:cust:frac_work', 'DESIGN.hpc.cust:frac_work')
prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
prob.model.connect('hpt:cool4:frac_P', 'DESIGN.hpt.cool4:frac_P')
prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')
prob.model.connect('lpt:cool2:frac_P', 'DESIGN.lpt.cool2:frac_P')
prob.model.connect('bypBld:frac_W', 'DESIGN.byp_bld.bypBld:frac_W')

# OFF DESIGN CASES
# pts = []
pts = ['OD1', 'OD2', 'OD3', 'OD4']

for i_OD, pt in enumerate(pts):
    ODpt = prob.model.add_subsystem(pt, CFM56(design=False))

    prob.model.connect('OD_alt', pt + '.fc.alt', src_indices=i_OD)
    prob.model.connect('OD_MN', pt + '.fc.MN', src_indices=i_OD)
    prob.model.connect('OD_Fn_target', pt + '.balance.rhs:FAR', src_indices=i_OD)
    prob.model.connect('OD_dTs', pt + '.fc.dTs', src_indices=i_OD)
    prob.model.connect('OD_cust_fracW', pt + '.hpc.cust:frac_W', src_indices=i_OD)

    prob.model.connect('inlet:ram_recovery', pt + '.inlet.ram_recovery')
    # prob.model.connect('splitter:BPR', pt+'.splitter.BPR')
    prob.model.connect('duct4:dPqP', pt + '.duct4.dPqP')
    prob.model.connect('duct6:dPqP', pt + '.duct6.dPqP')
    prob.model.connect('burner:dPqP', pt + '.burner.dPqP')
    prob.model.connect('duct11:dPqP', pt + '.duct11.dPqP')
    prob.model.connect('duct13:dPqP', pt + '.duct13.dPqP')
    prob.model.connect('core_nozz:Cv', pt + '.core_nozz.Cv')
    prob.model.connect('duct15:dPqP', pt + '.duct15.dPqP')
    prob.model.connect('byp_nozz:Cv', pt + '.byp_nozz.Cv')
    prob.model.connect('hp_shaft:HPX', pt + '.hp_shaft.HPX')

    prob.model.connect('hpc:cool1:frac_W', pt + '.hpc.cool1:frac_W')
    prob.model.connect('hpc:cool1:frac_P', pt + '.hpc.cool1:frac_P')
    prob.model.connect('hpc:cool1:frac_work', pt + '.hpc.cool1:frac_work')
    prob.model.connect('hpc:cool2:frac_W', pt + '.hpc.cool2:frac_W')
    prob.model.connect('hpc:cool2:frac_P', pt + '.hpc.cool2:frac_P')
    prob.model.connect('hpc:cool2:frac_work', pt + '.hpc.cool2:frac_work')
    prob.model.connect('bld3:cool3:frac_W', pt + '.bld3.cool3:frac_W')
    prob.model.connect('bld3:cool4:frac_W', pt + '.bld3.cool4:frac_W')
    # prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
    prob.model.connect('hpc:cust:frac_P', pt + '.hpc.cust:frac_P')
    prob.model.connect('hpc:cust:frac_work', pt + '.hpc.cust:frac_work')
    prob.model.connect('hpt:cool3:frac_P', pt + '.hpt.cool3:frac_P')
    prob.model.connect('hpt:cool4:frac_P', pt + '.hpt.cool4:frac_P')
    prob.model.connect('lpt:cool1:frac_P', pt + '.lpt.cool1:frac_P')
    prob.model.connect('lpt:cool2:frac_P', pt + '.lpt.cool2:frac_P')
    prob.model.connect('bypBld:frac_W', pt + '.byp_bld.bypBld:frac_W')

    prob.model.connect('DESIGN.fan.s_PR', pt + '.fan.s_PR')
    prob.model.connect('DESIGN.fan.s_Wc', pt + '.fan.s_Wc')
    prob.model.connect('DESIGN.fan.s_eff', pt + '.fan.s_eff')
    prob.model.connect('DESIGN.fan.s_Nc', pt + '.fan.s_Nc')
    prob.model.connect('DESIGN.lpc.s_PR', pt + '.lpc.s_PR')
    prob.model.connect('DESIGN.lpc.s_Wc', pt + '.lpc.s_Wc')
    prob.model.connect('DESIGN.lpc.s_eff', pt + '.lpc.s_eff')
    prob.model.connect('DESIGN.lpc.s_Nc', pt + '.lpc.s_Nc')
    prob.model.connect('DESIGN.hpc.s_PR', pt + '.hpc.s_PR')
    prob.model.connect('DESIGN.hpc.s_Wc', pt + '.hpc.s_Wc')
    prob.model.connect('DESIGN.hpc.s_eff', pt + '.hpc.s_eff')
    prob.model.connect('DESIGN.hpc.s_Nc', pt + '.hpc.s_Nc')
    prob.model.connect('DESIGN.hpt.s_PR', pt + '.hpt.s_PR')
    prob.model.connect('DESIGN.hpt.s_Wp', pt + '.hpt.s_Wp')
    prob.model.connect('DESIGN.hpt.s_eff', pt + '.hpt.s_eff')
    prob.model.connect('DESIGN.hpt.s_Np', pt + '.hpt.s_Np')
    prob.model.connect('DESIGN.lpt.s_PR', pt + '.lpt.s_PR')
    prob.model.connect('DESIGN.lpt.s_Wp', pt + '.lpt.s_Wp')
    prob.model.connect('DESIGN.lpt.s_eff', pt + '.lpt.s_eff')
    prob.model.connect('DESIGN.lpt.s_Np', pt + '.lpt.s_Np')

    prob.model.connect('DESIGN.core_nozz.Throat:stat:area', pt + '.balance.rhs:W')
    prob.model.connect('DESIGN.byp_nozz.Throat:stat:area', pt + '.balance.rhs:BPR')

    prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt + '.inlet.area')
    prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt + '.fan.area')
    prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt + '.splitter.area1')
    prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt + '.splitter.area2')
    prob.model.connect('DESIGN.duct4.Fl_O:stat:area', pt + '.duct4.area')
    prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt + '.lpc.area')
    prob.model.connect('DESIGN.duct6.Fl_O:stat:area', pt + '.duct6.area')
    prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt + '.hpc.area')
    prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt + '.bld3.area')
    prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt + '.burner.area')
    prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt + '.hpt.area')
    prob.model.connect('DESIGN.duct11.Fl_O:stat:area', pt + '.duct11.area')
    prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt + '.lpt.area')
    prob.model.connect('DESIGN.duct13.Fl_O:stat:area', pt + '.duct13.area')
    prob.model.connect('DESIGN.byp_bld.Fl_O:stat:area', pt + '.byp_bld.area')
    prob.model.connect('DESIGN.duct15.Fl_O:stat:area', pt + '.duct15.area')

prob.setup(check=False)

# initial guesses
prob['DESIGN.balance.FAR'] = 0.025
prob['DESIGN.balance.W'] = 100.
prob['DESIGN.balance.lpt_PR'] = 4.0
prob['DESIGN.balance.hpt_PR'] = 3.0
prob['DESIGN.fc.balance.Pt'] = 5.2
prob['DESIGN.fc.balance.Tt'] = 440.0

W_guesses = [300, 300, 700, 700]
for i, pt in enumerate(pts):
    # ADP and TOC guesses
    prob[pt + '.balance.FAR'] = 0.02467
    prob[pt + '.balance.W'] = W_guesses[i]
    prob[pt + '.balance.BPR'] = 5.105
    prob[pt + '.balance.lp_Nmech'] = 5000  # 4666.1
    prob[pt + '.balance.hp_Nmech'] = 15000  # 14705.7
    # prob[pt+'.fc.balance.Pt'] = 5.2
    # prob[pt+'.fc.balance.Tt'] = 440.0
    prob[pt + '.hpt.PR'] = 3.
    prob[pt + '.lpt.PR'] = 4.
    prob[pt + '.fan.map.RlineMap'] = 2.0
    prob[pt + '.lpc.map.RlineMap'] = 2.0
    prob[pt + '.hpc.map.RlineMap'] = 2.0

st = time.time()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
prob.run_model()
prob.model.DESIGN.list_outputs(residuals=True, residuals_tol=1e-2)

for pt in ['DESIGN'] + pts:
    viewer(prob, pt)

print()
print("time", time.time() - st)