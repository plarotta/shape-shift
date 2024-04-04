from simulate.simulation import Simulation

def test_simulation():
    '''checking that the simulation log is the correct size'''
    sim = Simulation()
    sim.run_simulation(sim_length=10, time_step=.001,log_k=3,mu_s=3,mu_k=3,floor_z_position=0, save=False)
    assert (len(sim.sim_log) == int(10/0.001/3)+1) and \
        all([(len(sim.sim_log[i]['masses']),len(sim.sim_log[i]['springs']),len([sim.sim_log[i]['speed']])) == (8,28,1) for i in sim.sim_log])