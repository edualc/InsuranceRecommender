import os
# import torch
import warnings
# from torch import nn
import numpy as np
import copy
import pandas as pd
import shutil

"""ParetoManager class for the MAMO framework.
  The ParetoManager keeps the pareto-front updated and saves the best-model, i.e pareto optimal models.
  Typical usage example:

  foo = ParetoManager()
  for e in no_epoch:
        foo.update_pareto_front()

  More precisely, the ParetoManager will save models if they are currently optimal,
  and it will also clean them as soon as they are not optimal anymore.
"""


class ParetoManager(object):
    """ ParetoManager class that is used to update the pareto-front and save bests models,
    while removing not optimal ones.
    This class handles the update of the pareto-front,
    the saving of the best models and the removing of non dominant models for the MAMO framework.
    """

    def __init__(self, sess, args, PATH='pareto_models/models/'):
        """Initialization of the class.

        Attributes:
        - PATH: path to store the models.
        - pareto_front : list of the pareto-front points

        - all_solutions: list of all the solutions tuples: (solution_metrics : list, solution_id: int).
                         Note: pareto optimal and dominated combined.

        - id_count: id counter to keep track the corresponding solutions with the good model update.
        """

        # Variables.
        # path to save the model.
        self.path = PATH

        # list for the pareto-front
        self._pareto_front = {1: [], 2: [], 3: [], 4: [], 5: []}
        # list of all the points
        self._all_solutions = {1: [], 2: [], 3: [], 4: [], 5: []}
        # id counter
        self.id_count = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        
        self.sess = sess
        self.args = args
        self.saver = None

    def __check_input_solution(self, solution):
        """A function that checks the input solution
        """
        if solution is None:
            raise TypeError(
                'Argument: the input solution must be a list representing a point.')
        # if not isinstance(solution, list):
            # raise TypeError(
            # 'The input solution should be a list repressenting a point!')
        # if len(solution) == 0:
            # raise ValueError(
            # 'Empty list was given as input, list should not be empty!')

    def __check_input_model(self, model):
        """A function that checks the input model
        """
        if model is None:
            raise TypeError(
                'Argument: model must be a model derived from pytorch.')
        # if not isinstance(model, nn.Module):
            # raise TypeError(
            # 'Argument: model must be a model derived from pytorch.')

    def _dominates(self, row, candidateRow):
        """A method that computes if row dominates candidateRow. It is the comparison method
        used to check if one solution dominates another, used for the computation of the add_solution method.

        Attributes:
        row: list representing a point - ex: [97, 23]
        candidateRow: list representing a point - ex: [55, 77]

        Output:
        -False if row does not dominates candidateRow
        -True if row does dominates candidateRow

        Example:
        self._dominates([2,3,7],[1,2,8]) => False
        """

        for i in range(len(row)):
            if(row[i] < candidateRow[i]):
                return False
        return True

    def _is_dominated(self, candidate_solution):
        """A method that checks if the candidate_solution is dominated or not by the solutions
        inside the current pareto front.

        Attributes:
        -candidate_solution: list representing a point - ex: [97,23]

        Output:
        -False - if candidate_solution is not dominated by any the solutions in the pareto front
        -True - if candidate_solution is dominated by a solution in the pareto front
        """
        for s, _ in self._pareto_front[self.K]:
            if self._dominates(s, candidate_solution):
                return True
        return False

    def add_solution(self, solution, saver, all_results_df, K):
        """A method that adds the solution to the current pareto front and maintains the
        pareto front, meaning it will remove any solution that may be dominated by adding
        the new solution.

        Attributes:
        -solution: list representing a point -ex: [97, 23]
        -model: Model which corresponds to the solution.
        -K: top K results
        """
        self.K = K
        self.saver = saver
        # Check the input solution and model
        self.__check_input_solution(solution)
        # self.__check_input_model(model)

        # append an id to the solution
        current_solution = (solution, self.id_count[self.K])
        self.id_count[self.K] += 1

        # append current solution to all solutions
        self._all_solutions[self.K].append(current_solution)

        # if current_solution[0] (list of points) is not dominated (not False),
        # and is not already in the pareto_front then it saves the current model
        # it also add the current_solution to the pareto_front and finally it clean the pareto_front
        # (previous solutions that are not solutions anymore are removed)
        if(current_solution[0] not in [x[0] for x in self._pareto_front[self.K]]):
            if not self._is_dominated(current_solution[0]):
                self._save_model(current_solution, all_results_df)
                self._pareto_front[self.K].append(current_solution)
                self._clean_pareto_front()
        return self._pareto_front[self.K], self._all_solutions[self.K], self.path_to_save_model

    def _clean_pareto_front(self):
        """A method to clean the pareto front.
        If a new solution is added to the pareto front, it should then be updated
        and previous outdated models removed from disk.

        """
        # print("in clean pareto front")
        # import code; code.interact(local=dict(globals(), **locals()))
        # copy the pareto front because he will remove points from it.
        pareto_front_copy = copy.deepcopy(self._pareto_front[self.K])

        for current_solution in pareto_front_copy:
            self._pareto_front[self.K].remove(current_solution)

            # If the solution is still dominant, we add it to solution set
            # Else Remove/delete it from disk.
            if not self._is_dominated(current_solution[0]):
                self._pareto_front[self.K].append(current_solution)

            else:
                self._remove_model(current_solution)

    def _save_model(self, solution, all_results_df):
        """A method to save the model with a given string name.
        This method will save the model under (by default):
        '/paretoFile/<ModelName>/<ModelName>_<Epoch>'

        Where <ModelName> is the name of the pytorch derived model and <Epoch> is the epoch
        at the moment the models was stored.

        Attributes:
        - model: model that is currently trained
        """
        subdir_path = self._make_subpath()

        # retrieve the name with the coresponding identifier
        self.path_to_save_model = os.path.join(
            self.path, subdir_path)

        self.path_to_save_solution = os.path.join(
            self.path, subdir_path, self._solution_to_str_rep(solution))

        if not os.path.isdir(self.path_to_save_solution):
            os.makedirs(self.path_to_save_solution)
        # ignore the warning and save the model
        warnings.filterwarnings('ignore')
        all_results_df.to_csv(self.path_to_save_solution +
                              '/results.csv', index=False)

        # self.saver.save(self.sess, self.path_to_save_model + '/model.ckpt')

    def _remove_model(self, solution):
        """A method to remove the non dominant models (models that are not in the pareto front anymore).
        """
        # retrieve the complete name
        subdir_path = self._make_subpath()
        self.path_to_save_solution = os.path.join(
            self.path, subdir_path, self._solution_to_str_rep(solution))

        # check if the name is correct/exists.
        if os.path.exists(self.path_to_save_solution):
            # delete it.
            shutil.rmtree(self.path_to_save_solution)

    def _solution_to_str_rep(self, solution):
        """A method to format the string representation name, to save the models and also to
        be able to retrieve it, such that it can be removed later on.
        """
        s, s_id = solution
        return ('id_%s_val_metrics_') % s_id + '_'.join(['%.4f']*len(s)) % tuple(s)

    def _make_subpath(self):
        loss_names = ''
        for loss in self.args.losses_names:
            loss_names = loss_names + loss + '_'
        # make a directory with model params if not exist.
        subdir_path = 'batch' + str(self.args.batch_size) + '_' + \
            str(self.args.method) + '_' + \
            'lr' + str(self.args.lr) + '_' + \
            'lambda' + str(self.args.lambda_value) + '_' + \
            'opt' + str(self.args.optimizer_method) + '_' + \
            'obj' + str(loss_names) + \
            'split' + str(self.args.split_num)
        return subdir_path
