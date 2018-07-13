import pygmo as pg
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)  
sys.setdefaultencoding('utf-8')

seed = 42

import pandas as pd

problems_df = pd.DataFrame.from_csv('problems.csv', sep=';')
problems_df['Best Fitness'] = 0.0
problems_df['Number of Evaluations'] = 0
problems_df['Closeness to Global Minimal'] = 0.0
problems_df['log'] = ''

algo = pg.algorithm(pg.sga(gen       = 100000, 
                           cr        = 0.9,
                           m         = 0.2,
                           param_m   = 0.2,
                           param_s   = 10,
                           crossover = 'single',
                           mutation  = 'gaussian',
                           selection = 'tournament',
                           seed      = 42))


problem_id = 5
for problem_id in problems_df.index:
    prob = pg.problem(pg.cec2014(dim=2, prob_id=problem_id))

    pop = pg.population(prob=prob, size = 100)
    algo.set_verbosity(1)
    pop = algo.evolve(pop)
    uda = algo.extract(pg.sga)

    problems_df['Best Fitness'][problem_id] = pop.get_f()[pop.best_idx()]
    problems_df['Closeness to Global Minimal'][problem_id] = problems_df['Global Minimal'][problem_id] / problems_df['Best Fitness'][problem_id]

    log = uda.get_log()
    fitness_over_time = [x[2] for x in log]
    problems_df['log'][problem_id] = json.dumps(log)
    nb_evals = [x[1] for x in log]
    problems_df['Number of Evaluations'][problem_id] = nb_evals[-1]


    plt.plot(fitness_over_time)
    plt.xticks(np.arange(len(nb_evals))[::len(nb_evals)/8], nb_evals[::len(nb_evals)/8], rotation = 30)

    plt.axhline(y=problems_df['Global Minimal'][problem_id], color='r', linestyle='--')
    plt.xlabel('Number of function evaluations')
    plt.ylabel('Fitness eval')
    plt.title('Problem %d: %s'%(problem_id, problems_df['Function Name'][problem_id]))
    plt.savefig('logs/figures/Figure-prob'+problem_id+'.eps', format = 'eps')
    plt.show(block = False)

    pd.options.display.float_format = '{:,.8f}'.format
    problems_df[filter(lambda column: column != 'log', problems_df.columns)]

problems_df.to_csv('logs/results-preliminar.csv')