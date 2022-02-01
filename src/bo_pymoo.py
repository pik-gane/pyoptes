import numpy as np
from tqdm import tqdm

from pyoptes import set_seed
from pyoptes.optimization.budget_allocation import target_function as f

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

import matplotlib.pyplot as plt


class ProblemSIModel(ElementwiseProblem):
    def __init__(self, n_var=120,
                 n_obj=1,
                 n_constr=1,    # 1 constraint
                 xl=0.0,
                 xu=120.0):
        """
        @param n_var: int, number of input variables
        @param n_obj: int, number of objectives that are to be optimized
        @param n_constr: int, number of constraints on the optimization
        @param xl: float, lower bound of input variables
        @param xu: float, upper bound of input variables
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        set_seed(1)
        # TODO even only creating an object of f creates an error: TypeError: cannot pickle 'module' object
        self.f = f
        self.f.prepare(n_nodes=n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x)
        # print(type(x), np.shape(x))
        out["F"] = x.sum() #self.f.evaluate(x)
        out["G"] = out["F"].sum() - 120.0
        # print(out["G"])


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]
        # print(out["G"])


if __name__ == '__main__':
    # n_nodes = 120
    # f.prepare(n_nodes=n_nodes)
    # print('d')

    problem = ProblemSIModel()
    # problem = MyProblem()

    algorithm = NSGA2(
        pop_size=4,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True)

    termination = get_termination("n_gen", 40)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F

    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()

    # a = SiModel(n_var=10)
    # print(a.xl)
    # print(a.xu)