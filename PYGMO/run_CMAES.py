import pygmo as pg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import json

def mkdir_p(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

dimensions = [10, 30]
nb_runs  = 51
nb_population = 50


problems_to_be_eval = [1, 2, 6, 7, 9, 14]
 
best_runs = {}
test_run = {}
test = 9

folder = 'logs/cmaesBee'+str(test)
mkdir_p(folder)
mkdir_p(folder+'/figures')

params_to_str = "CMAES WITH POP 50 and BEE COLONIES"

for dim in dimensions:

    #nb_population = int(3*np.log(dim) + 4)

    problems_df = pd.read_csv('problems.csv', sep=';')
    problems_df.index = problems_df['Index']
    problems_df.drop('Index', axis=1, inplace=True)
    problems_df = problems_df[problems_df.index.isin(problems_to_be_eval)]
    problems_df['Best Fitness run'] = 0.0
    problems_df['Worst Fitness run'] = 0.0
    problems_df['Mean Fitness run'] = 0.0
    problems_df['Median Fitness run'] = 0.0
    problems_df['STD Fitness run'] = 0.0
    problems_df['Number of Evaluations - Worst run'] = 0
    problems_df['Number of Evaluations - Mean run'] = 0
    problems_df['Number of Evaluations - Best run'] = 0
    problems_df['SuccessRate'] = 0.0
    problems_df['log'] = ''

    for problem_id in problems_to_be_eval:
        try:
            prob = pg.problem(pg.cec2014(dim=dim, prob_id=problem_id))
            try:
                best_runs[problem_id]
            except KeyError:
                best_runs[problem_id] = {}
        except ValueError:
            continue

        print('Started problem: %d'%problem_id)

        max_function_eval = 10000*dim
        max_number_generations = max_function_eval//nb_population
        nb_gen = int(0.7 * max_number_generations)
        nb_gen_pso = max_number_generations - nb_gen

        algo_cmaes = pg.algorithm(pg.cmaes(gen = nb_gen, ftol=1e-8, xtol=1e-8))
        algo_pso   = pg.algorithm(pg.bee_colony(gen = nb_gen_pso))
        all_fitness = []
        all_nb_evals = []
        all_logs = {}

        for run in range(nb_runs):
            pop = pg.population(prob=prob, size = nb_population)
            algo_cmaes.set_verbosity(1)
            #pop = algo_pso.evolve(pop)
            pop = algo_cmaes.evolve(pop)
            uda = algo_cmaes.extract(pg.cmaes)

            log = uda.get_log()
            fitness_over_time = [x[2] for x in log]
            nb_evals = [x[1] for x in log]
            
            if len(fitness_over_time) > 0:
                all_fitness += [fitness_over_time]
                all_nb_evals += [nb_evals]
                all_logs[run] = log

        problems_df.at[problem_id ,'log'] = deepcopy(json.dumps(all_logs))

        best_fitness_idx = np.argmin(list(map(lambda x: x[-1], all_fitness)))
        worst_fitness_idx = np.argmax(list(map(lambda x: x[-1], all_fitness)))

        min_eval = all_nb_evals[best_fitness_idx][-1]
        max_eval = all_nb_evals[worst_fitness_idx][-1]
        mean_eval = np.mean(list(map(lambda x: x[-1], all_nb_evals)))

        problems_df.at[problem_id, 'Number of Evaluations - Best run'] = deepcopy(min_eval)
        problems_df.at[problem_id, 'Number of Evaluations - Worst run'] = deepcopy(max_eval)
        problems_df.at[problem_id, 'Number of Evaluations - Mean run'] = deepcopy(mean_eval)

        problems_df.at[problem_id, 'Best Fitness run']  = deepcopy(all_fitness[best_fitness_idx][-1]) - problems_df['Global Minimal'][problem_id]
        problems_df.at[problem_id, 'Worst Fitness run'] = deepcopy(all_fitness[worst_fitness_idx][-1]) - problems_df['Global Minimal'][problem_id]
        problems_df.at[problem_id, 'Mean Fitness run']  = deepcopy(np.mean(list(map(lambda x: x[-1], all_fitness)))) - problems_df['Global Minimal'][problem_id]
        problems_df.at[problem_id, 'Median Fitness run']  = deepcopy(np.median(list(map(lambda x: x[-1], all_fitness)))) - problems_df['Global Minimal'][problem_id]
        problems_df.at[problem_id, 'STD Fitness run'] = deepcopy(np.std(list(map(lambda x: x[-1] - problems_df['Global Minimal'][problem_id], all_fitness))))

        problems_df.at[problem_id, 'SuccessRate'] = deepcopy(np.sum(list(map(lambda x: x[-1] - problems_df['Global Minimal'][problem_id] < 1E-8, all_fitness)))/51)

        plt.figure()
        plt.axhline(y=problems_df['Global Minimal'][problem_id], color='r', linestyle='--')
        plt.semilogy(problems_df.at[problem_id, 'Number of Evaluations - Best run'], 
                    problems_df.at[problem_id, 'Best Fitness run'] )
        plt.xticks(np.arange(len(nb_evals))[::(len(nb_evals)//8 or 1)], nb_evals[::(len(nb_evals)//8 or 1)], rotation = 25)

        plt.xlabel('Number of function evaluations')
        plt.ylabel('Best Fitness eval')
        plt.title('Problem %d at dim %d: %s\nBest Result: %.8f'%(problem_id, dim, problems_df['Function Name'][problem_id], problems_df['Best Fitness run'][problem_id]))
        plt.savefig(folder+'/figures/Figure-prob'+str(problem_id)+'-dim'+str(dim)+'.eps', format = 'eps')

        print('Ended, with results:')
        print('     - Best fitness: %.8f for %.8f'%(problems_df['Best Fitness run'][problem_id], problems_df['Global Minimal'][problem_id]))
        print('     - Number of Evaluations: %d'%problems_df['Number of Evaluations - Best run'][problem_id])
        print('-------------------------------------------------------------------')
        best_runs[problem_id][params_to_str] = problems_df.at[problem_id, 'Best Fitness run']
    problems_df.to_csv(folder+'/results-dim' + str(dim) + '.csv', sep = ';')

test_run[params_to_str] = test
test += 1


print("---------------------------------")
min_run = {}
for problem_id in problems_to_be_eval:
    min_run[problem_id] = min(best_runs[problem_id], key = best_runs[problem_id].get)
    print('Best fitness: %.8f on test %d at %s'%(best_runs[problem_id][min_run[problem_id]], test_run[min_run[problem_id]],  min_run[problem_id]))


print("---------------------------------")

results_file = open('logs/all_results.txt', mode = 'w')
results_file.write(json.dumps(best_runs))
results_file.write(json.dumps(test_run))
results_file.close()


for dim in dimensions:
    for problem_id in problems_to_be_eval:
        df = pd.read_csv(folder+'/results-dim' + str(dim) + '.csv', sep = ';')
        df.index = df['Index']
        df.drop('Index', axis=1, inplace=True)
        max_fes = 10000*dim
        AG_nb_FC = np.array([0.0, 0.0001*max_fes, 0.01*max_fes] + [x*max_fes for x in np.linspace(0.1, 1.0, num=10)], dtype=np.int)
        problem_id_logs = json.loads(df['log'][problem_id])
        error_evolution_df = pd.DataFrame(AG_nb_FC)
        error_evolution_df.columns = ['AG_#FC_'+str(dim)]
        for run in np.arange(nb_runs):
            error_evolution_index = 0
            error_evolution_df['Run'+str(run+1)] = 0.0
            for gen in problem_id_logs[str(run)]:
                error_evolution_df.at[error_evolution_index, 'Run'+str(run+1)] = gen[2] - df['Global Minimal'][problem_id]
                if gen[1] > error_evolution_df['AG_#FC_'+str(dim)][error_evolution_index]:
                    error_evolution_index = error_evolution_index + 1

        error_evolution_df.to_csv(folder+'/fevals-dim' + str(dim) + '-prob' + str(problem_id) +'.csv', sep = ';')