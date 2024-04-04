from evolve.ga import Evolution
import numpy as np

def test_evolution_eval():
    np.random.seed(42)
    eve = Evolution()
    m0 = np.copy(eve.masses)
    spring_constants = eve.initialize_population(1, eve.springs)
    res = eve.eval_springs(eve.masses, eve.springs, spring_constants[0], 4, 0.001)
    expected = 4.5905
    assert np.all(np.isclose(m0,eve.masses)), 'problem with rehome. masses were not reset'
    assert np.isclose(res,expected), 'problem with eval. did not get expected value'

def test_mutate():
    np.random.seed(42)
    eve = Evolution()
    m0 = np.copy(eve.masses)
    spring_constants = eve.initialize_population(1, eve.springs)

    m_new_f, _, _ = eve.mutate_morphology(m0, eve.springs, spring_constants[0],p_fatten=1)
    assert m0.shape != m_new_f.shape, 'problem with fatten operation of mutate morphology. masses did not change'

    m_new_s, _, _ = eve.mutate_morphology(m0, eve.springs, spring_constants[0],p_fatten=0)
    assert m0.shape != m_new_s.shape, 'problem with slim operation of mutate morphology. masses did not change'





