#include <iostream>
#include <cmath>
#include <list>
#include <vector>
#include <ctime>
#include <random>
#include <algorithm> 

#define INF 1.0e99
#define EPS 1.0e-14
#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

using namespace std;

class Population { 
    public:
        int coding_length;
        int survivor_size;
        int offspring_size;
        int all_pop_size;
        float mutation_probability;
        float crossover_probability;
        double *fitness_list;
        double *pop_list;
        void (*fitness_function)(double *, double *,int,int,int);
        default_random_engine generator;

    Population(void (*f)(double *, double *,int,int,int)) {     
        coding_length = 2;
        survivor_size = 100;
        offspring_size = 100;
        all_pop_size = survivor_size + offspring_size;
        mutation_probability = 0.05;
        crossover_probability = 0.8;
        fitness_function = f;

        fitness_list = new double[all_pop_size];
        pop_list = new double[all_pop_size * coding_length];

        generator.seed(time(NULL));
        generate();
    };

    Population(void (*f)(double *, double *,int,int,int), int dim, int pop_size, int children_size) {
        coding_length = dim;
        survivor_size = pop_size;
        offspring_size = children_size;
        all_pop_size = survivor_size + offspring_size;
        mutation_probability = 0.05;
        crossover_probability = 0.8;
        fitness_function = f;

        fitness_list = new double[all_pop_size];
        pop_list = new double[all_pop_size * coding_length];

        generator.seed(time(NULL));
        generate();
    };

    Population(void (*f)(double *, double *,int,int,int), int dim, int pop_size, int children_size, float m_prob, float c_prob) {
        coding_length = dim;
        survivor_size = pop_size;
        offspring_size = children_size;
        all_pop_size = survivor_size + offspring_size;
        mutation_probability = m_prob;
        crossover_probability = c_prob;
        fitness_function = f;
        fitness_list = new double[all_pop_size];
        pop_list = new double[all_pop_size * coding_length];
        generator.seed(time(NULL));
        generate();
    };

    void generate() {
        uniform_real_distribution<double> distribution(-10, 10);
        for (int j = 0; j < survivor_size * coding_length; j+= coding_length) {
             cout << "First Gen "<< j << ":";
            for (int k = 0; k < coding_length; k++){ 
                pop_list[j + k] = distribution(generator);
                 cout << " - "<< j + k << ": " << pop_list[j + k];
            }
            cout << "\n";
        };
        compute_fitness(survivor_size, 1);
    };
    
    void compute_fitness(int total_pop, int func_num) {

        fitness_function(pop_list, fitness_list, coding_length, total_pop + 1, func_num);
    };

    void tournament(int nb_challengers, int total_pop, int * score_array) {
        srand(time(NULL));
        short score;
        short arrayIndex;

        for (int i = 0; i < total_pop; i++) {
            score = 0;
            for (int j = 0; j < nb_challengers - 1; j++) {
                arrayIndex = rand() % (survivor_size / nb_challengers);
                if (abs(fitness_list[i]) < abs(fitness_list[arrayIndex])) {
                    score += 3;
                }
                else if (abs(fitness_list[i]) == abs(fitness_list[arrayIndex])) {
                    score += 1;
                };
                arrayIndex += survivor_size / nb_challengers;
            };
            score_array[i] = score;
        };
    };

    void roullete(double *chosen, const int total_pop, unsigned int size) {
        //int total_pop;
        double fitness_sum = 0;
        double chance;
        unsigned int n = 0;
        int pop_index = 0;
        double *prob_array = new double[total_pop];
        uniform_real_distribution<double> distribution(0, 1.0);

        compute_fitness(total_pop, 1);
        // Get sum of all fitness
        for (int i = 0; i < total_pop; i++) {
            fitness_sum += abs(fitness_list[i]);
            //cout << "index " << i << " " << fitness_sum << "\n";
        };
        //cout << "fitn " << fitness_sum << "\n";
        //cout << fitness_sum << " "    << abs(fitness_list[0]);
        double prob_array_sum = 0;
        for (int i = 0; i < total_pop; i++) {
            prob_array[i] = 1 - abs(fitness_list[i])/fitness_sum;
            prob_array_sum += prob_array[i];
        };

        for (int i = 0; i < total_pop; i++) { 
            prob_array[i] = prob_array[i]/prob_array_sum;
            //cout << "i" << i << ": " << prob_array[i] << " " << fitness_list[i] << "  \n";
        };
/*      

        double offset = 0;
        for (int i = 0; n < size; i++) { 
            chance = distribution(generator);
            pop_index = i % total_pop;
            if (chance < prob_array[pop_index]) {

            }
        };*/


        // Choose based on fitness probability which individuals will be copied
        n = 0;
        for (int i = 0; n < size * coding_length; i++) {
            pop_index = (i % total_pop);
            chance = distribution(generator);

            if (chance < prob_array[pop_index]) {
                cout << "chosen: " << pop_index*coding_length << "\n";
                
                for (int k = 0; k < coding_length; k++) {
                    chosen[n + k] = pop_list[pop_index*coding_length + k];
                    cout << n + k << " - " << ": " << pop_index*coding_length + k << " " << chosen[n + k] << "\n";
                };
                n = n + coding_length;
            };
        };         
        cout << "\n";
    };

    void mutation(double *children, double *mutatedChildren, short size) {
        short arrayIndex;
        double chance;
        uniform_real_distribution<double> distribution(-2.0, 2.0);

        for (int individual = 0; individual < size * coding_length; individual = individual + coding_length) {
            chance = double(rand())/RAND_MAX;
            for (int i = 0; i < coding_length; i++) {
                mutatedChildren[individual + i] = children[individual + i];
            };
            if (mutation_probability > chance) {
                arrayIndex = short(rand() % coding_length);
                mutatedChildren[individual + arrayIndex] = mutatedChildren[individual + arrayIndex] + 
                                                distribution(generator);
            };
        };

    }

    void crossover(double* chosen, double* children, int chosenSize) {
        // Children must be size of offspring_size
        short crossover_point;
        double chance;
        int firstParentIndex;
        int secondParentIndex;
        int secondChildIndex;
        uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int childIndex = 0; childIndex < offspring_size/2; childIndex++) {
            chance = distribution(generator);
            secondChildIndex = childIndex + offspring_size/2;
            if (chance < crossover_probability) {
                crossover_point = short(rand() % coding_length);
                firstParentIndex = childIndex % chosenSize;
                secondParentIndex = (firstParentIndex == chosenSize - 1) ? 0: firstParentIndex + 1;
                // cout << "chid1: " << childIndex << " child2: " << secondChildIndex << "\n";
                // cout << "   First" << firstParentIndex << ": " << chosen[firstParentIndex] << " - " << chosen[firstParentIndex + 1];
                // cout << "   Second" << secondParentIndex << ": " << chosen[secondParentIndex] << " - " << chosen[secondParentIndex + 1];
                // cout<< "\n";
                for (int i = 0; i < crossover_point; i++) {
//                    children[childIndex + i] = chosen[firstParentIndex + i];
                    children[secondChildIndex + i] = (chosen[firstParentIndex + i] - chosen[secondParentIndex + i]);
//                    children[secondChildIndex + i] = chosen[secondParentIndex + i];

                };
                for (int i = crossover_point; i < coding_length; i++) {

                    children[childIndex + i] = chosen[secondParentIndex + i] - chosen[firstParentIndex + i];
//                    children[childIndex + i] = chosen[secondParentIndex + i];
//                    children[secondChildIndex + i] = chosen[firstParentIndex + i];
                    children[secondChildIndex + i] = (chosen[firstParentIndex + i] + chosen[firstParentIndex + i]) / 2;
                };
            }
            else {
                firstParentIndex = childIndex % chosenSize;
                secondParentIndex = (firstParentIndex == chosenSize - 1) ? 0: firstParentIndex + 1;
                for (int i = 0; i < coding_length; i++) {
                    children[childIndex + i] = chosen[firstParentIndex + i];

                    children[secondChildIndex + i] = chosen[secondParentIndex + i];

                };
            };
        };

    };
};


//void cec14_test_func(double *x, double *f, int nx, int mx,int func_num) {
//}
void cec14_test_func(double *, double *, int, int, int);
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;

int main() {
    srand(time(NULL));
    const int survivor_size = 10;
    const int offspring_size = 10;
    const int coding_length = 2;
    const int func_number = 1;
    const int gen_number = 5;
    const float mutation_probability = 0.005;
    const float crossover_probability = 1.0;
    double chosen[survivor_size * coding_length] = {0};
    double mutated[offspring_size * coding_length] = {0};
    double children[offspring_size * coding_length] = {0};
    double all_pop[(survivor_size + offspring_size) * coding_length] = {0};

    Population pop(&cec14_test_func, coding_length, survivor_size, offspring_size, mutation_probability, crossover_probability);
    pop.compute_fitness(survivor_size, func_number);

    for (int i = 0; i < gen_number; i++) {
        cout << "--------------ROULETTE--------------\n";
        pop.roullete(chosen, pop.survivor_size, offspring_size);
        cout << "--------------CROSSOVER--------------\n";
        pop.crossover(chosen, children, offspring_size);
        cout << "--------------MUTATION--------------\n";
        pop.mutation(children, mutated, offspring_size);
        for (int j = survivor_size; 
                 j < (survivor_size + offspring_size) * coding_length; 
                 j += coding_length) {
            cout << "Offspring" << j << ":";
            for (int k = 0; k < coding_length; k++){ 
                pop.pop_list[j + k] = mutated[j + k - survivor_size];
                //pop.pop_list[j + k] = chosen[j + k - survivor_size];
                cout << " - " << pop.pop_list[j + k];
            }
            cout << "\n";
        };
        cout << "--------------FITNESS--------------\n";
        pop.compute_fitness(survivor_size + offspring_size, func_number);
        cout << "--------------ROULETTE FOR SURVIVAL--------------\n";
        pop.roullete(chosen, survivor_size + offspring_size, survivor_size);

        for (int j = 0; 
                 j < survivor_size * coding_length; 
                 j += coding_length) {
            cout << "Survived"<< j << ":";
            for (int k = 0; k < coding_length; k++){ 
                pop.pop_list[j + k] = chosen[j + k];
                cout << " - " << pop.pop_list[j + k];
            }
            cout << " / Fitness = " << pop.fitness_list[j];
            cout << "\n";
        };
        //if (i % 999 == 0){
        cout << "================================\n";
        cout << "END of GEN: " << i << "\n";
        cout << "Min: " << *min_element(pop.fitness_list, pop.fitness_list + survivor_size) << "\n";
        cout << "Max: " << *max_element(pop.fitness_list, pop.fitness_list + survivor_size) << "\n";
        cout << "================================\n";
        //}

    };
}
