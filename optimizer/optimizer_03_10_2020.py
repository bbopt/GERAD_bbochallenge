"""
ORTHO MADS 2n or n+1, with sgtelib search, no NM search, LH searches, and mesh initialization r0.001
Take into account the logit/log scaling of the variables for supposed
better precision for surrogate models (and separe logit from log)
It seems to have a warning error.


THIS STRATEGY RETURNED A SCORE OF:
    -local benchmarks: 97.348803



/home/saloludo/.local/lib/python3.8/site-packages/bayesmark/signatures.py:85: RuntimeWarning: Signature diverged on SVM_wine_nll betwen [0.65543451 0.67458882 0.64187048 0.6529983  0.65068383] and [0.65808769 0.68343106 0.659564   0.65425
534 0.66650674]
  warnings.warn(
Signature errors:
                       0             1         2         3         4           max
SVM_wine_nll    0.002653  8.842240e-03  0.017694  0.001257  0.015823  1.769352e-02
SVM_boston_mse  0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000e+00
SVM_boston_mae  0.000000  8.881784e-16  0.000000  0.000000  0.000000  8.881784e-16
DT_wine_acc     0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000e+00
SVM_wine_acc    0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000e+00
DT_boston_mae   0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000e+00
DT_boston_mse   0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000e+00
DT_wine_nll     0.000000  0.000000e+00  0.000000  0.000000  0.000000  0.000000e+00
max             0.002653  8.842240e-03  0.017694  0.001257  0.015823  1.769352e-02
{"exp-anal sig errors": {"SVM_wine_nll": {"0": 0.002653176054346451, "1": 0.008842239886381886, "2": 0.01769352086594056, "3": 0.0012570440036353547, "4": 0.01582291269931746, "max": 0.01769352086594056}, "SVM_boston_mse": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "max": 0.0}, "SVM_boston_mae": {"0": 0.0, "1": 8.881784197001252e-16, "2": 0.0, "3": 0.0, "4": 0.0, "max": 8.881784197001252e-16}, "DT_wine_acc": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "max": 0.0}, "SVM_wine_acc": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "max": 0.0}, "DT_boston_mae": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "max": 0.0}, "DT_boston_mse": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "max": 0.0}, "DT_wine_nll": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "max": 0.0}, "max": {"0": 0.002653176054346451, "1": 0.008842239886381886, "2": 0.01769352086594056, "3": 0.0012570440036353547, "4": 0.01582291269931746, "max": 0.01769352086594056}}}
Scores by problem (JSON):

{"_visible_to_opt": {"nomadsolution_x.x.x_327a886": {"DT_boston_mae": -0.07094909758781386, "DT_boston_mse": -0.02075312804891457, "DT_wine_acc": -0.02966101694669409, "DT_wine_nll": -0.013588778076597998, "SVM_boston_mae": -0.020031349308752784, "SVM_boston_mse": -0.04398631303518624, "SVM_wine_acc": 0.1128404669416296, "SVM_wine_nll": 0.29822500878845704}}}


median score @ 16:
optimizer
nomadsolution_x.x.x_327a886   -0.476567
mean score @ 16:
optimizer
nomadsolution_x.x.x_327a886    0.026512
Final scores (JSON):

{"_visible_to_opt": {"nomadsolution_x.x.x_327a886": 0.02651197409076589}}


normed mean score @ 16:
optimizer
nomadsolution_x.x.x_327a886    0.263234
Scores by problem (JSON):

{"generalization": {"nomadsolution_x.x.x_327a886": {"DT_boston_mae": 0.1756285465766117, "DT_boston_mse": 0.2222203490215274, "DT_wine_acc": 0.10526315789661904, "DT_wine_nll": 1.0, "SVM_boston_mae": 0.00014104959146659656, "SVM_boston_mse": -0.001215203639188563, "SVM_wine_acc": 1.998401444285349e-11, "SVM_wine_nll": 1.0}}}


median score @ 16:
optimizer
nomadsolution_x.x.x_327a886    1.0
mean score @ 16:
optimizer
nomadsolution_x.x.x_327a886    0.312755
Final scores (JSON):

{"generalization": {"nomadsolution_x.x.x_327a886": 0.31275473743337756}}


normed mean score @ 16:
optimizer
nomadsolution_x.x.x_327a886    2.666352
--------------------
Final score `100 x (1-loss)` for leaderboard:
optimizer
nomadsolution_x.x.x_327a886    97.348803

    - test serveurs : 83.691

"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import logit, expit
from nomadsolver import optimize as nomad_solve

#  import threading
#  import queue
import multiprocessing

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

# Sklearn prefers str to unicode:
DTYPE_MAP = {"real": float, "int": int, "bool": bool, "cat": str, "ordinal": str}

class PyNomadOptimizer(AbstractOptimizer):

    primary_import = None#"PyNomad"

    def __init__(self, api_config, random=np_util.random, n_initial_points=0):
        """Build wrapper class to use an optimizer in benchmark

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        n_initial_points : int
            Number of points to sample randomly before actual Nomad optimization
        """
        AbstractOptimizer.__init__(self, api_config)

        self.api_config = api_config

        print(api_config)

        main_params = PyNomadOptimizer.get_nomad_dimensions(api_config)

        # NB these params will have to be tuned according to the number of variables,
        # and their types
        #  params = [main_params[0], #main_params[1], main_params[2],
        #            'BB_OUTPUT_TYPE OBJ',
        #            'MAX_BB_EVAL ' + str(8 * 16),
        #            'BB_MAX_BLOCK_SIZE 8',
        #            'MODEL_SEARCH SGTELIB',
        #            'SGTELIB_MODEL_CANDIDATES_NB 8',
        #            'SGTELIB_MODEL_TRIALS 5',
        #            'MODEL_EVAL_SORT no',
        #            'DIRECTION_TYPE ORTHO 2N',
        #            'SPECULATIVE_SEARCH no',
        #            'LH_SEARCH 8 0',
        #            'OPPORTUNISTIC_EVAL false',
        #            'NM_SEARCH false'] #,'PERIODIC_VARIABLE 0']

        params = [main_params[0], #main_params[1], main_params[2],
                  'BB_OUTPUT_TYPE OBJ',
                  'MAX_BB_EVAL ' + str(8 * 16),
                  'BB_MAX_BLOCK_SIZE 8',
                  'MODEL_SEARCH SGTELIB',
                  'SGTELIB_MODEL_CANDIDATES_NB 8',
                  'SGTELIB_MODEL_TRIALS 5',
                  'MODEL_EVAL_SORT no',
                  'SPECULATIVE_SEARCH no',
                  'LH_SEARCH 8 0',
                  'OPPORTUNISTIC_EVAL true',
                  'OPPORTUNISTIC_LH false',
                  'NM_SEARCH false',
                  'INITIAL_MESH_SIZE r0.001'] #,'PERIODIC_VARIABLE 0'] 

        # fine tune parameters according to the number of variables
        dimensions_pb = len(self.api_config.keys())

        #  Direction type and intensification for dimension 3 to 9
        #   dim       2n         n+1       strategy
        #     3        6           4       2n + intens. 4 => 8
        #     4        8           5       2n => 8
        #     5       10           6       n+1 + intens. 2 => 8
        #     6       12           7       n+1 + intens. 1 => 8
        #     7       14           8       n+1 => 8
        #     8       16           9       2n => 2*8
        #     9       18          10       n+1 + intens. 6 => 2*8

        # choose direction type according to the type of the dimensions of the problem
        if dimensions_pb == 3 or dimensions_pb == 4 or dimensions_pb == 8:
            params.append('DIRECTION_TYPE ORTHO 2N')
        else:
            params.append('DIRECTION_TYPE ORTHO N+1 UNI')

        # intensification for some direction type and dimensions
        if dimensions_pb == 3:
            params.append('MAX_EVAL_INTENSIFICATION 4')
        if dimensions_pb == 5:
            params.append('MAX_EVAL_INTENSIFICATION 2')
        if dimensions_pb == 6:
            params.append('MAX_EVAL_INTENSIFICATION 1')
        if dimensions_pb == 9:
            params.append('MAX_EVAL_INTENSIFICATION 6')

        # deal with categorical variables
        if len(main_params[3]) > 0:
            instruction = 'PERIODIC_VARIABLE '
            for var_index in main_params[3]:
                instruction += str(var_index) + ' '
            params.append(instruction)

        self.round_to_values = main_params[4]

        # lower and upper bounds are given
        lb = main_params[1]
        ub = main_params[2]

        x0 = [] # TODO choose x0 or LHS with n_initial_points

        # TODO analyze the types of the inputs to fill at maximum nomad bb blocks

        # queues to communicate between threads
        #  self.inputs_queue = queue.Queue()
        #  self.outputs_queue = queue.Queue()
        self.inputs_queue = multiprocessing.JoinableQueue()
        self.outputs_queue = multiprocessing.JoinableQueue()

        # random stuff
        self.random = random

        # counter to deal with number of iterations: needed to properly kill the daemon thread
        self.n_iters = 0

        # list to keep candidates for an evaluation
        self.stored_candidates = list()

        # start background thread
        #  self.nomad_thread = threading.Thread(target=nomad_solve, args=(self.bb_fct, x0, lb, ub, params,), daemon=True)
        #  self.nomad_thread.start()
        self.nomad_process = multiprocessing.Process(target=nomad_solve, args=(self.bb_fct, x0, lb, ub, params,))
        self.nomad_process.start()


    @staticmethod
    def get_nomad_dimensions(api_config):
        """Help routine to setup PyNomad search space in constructor

            Take api_config in argument so this can be static
        """
        # there is equally a JointSpace method which could be used (see pysot)

        # take a look for argument; should be sorted normally
        # just in case we do it
        param_list = sorted(api_config.keys())

        bb_input_type_string = 'BB_INPUT_TYPE ( '
        #  bb_lower_bound_string = 'LOWER_BOUND ('
        #  bb_upper_bound_string = 'UPPER_BOUND ('
        lb = []
        ub = []
        tol = 10**(-7)

        periodic_var_indexes = list()
        counter_periodic_vars = 0

        round_to_values = {}

        for param_name in param_list:
            param_config = api_config[param_name]
            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # some parameters can be real or integer and take a finite set of values
            # TODO : how to deal with them
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
                round_to_values[param_name] = interp1d(
                    param_values, param_values, kind="nearest", fill_value="extrapolate"
                )

            if param_type == "int":
                bb_input_type_string += 'I '
                #  bb_lower_bound_string += ' ' + str(param_range[0])
                #  bb_upper_bound_string += ' ' + str(param_range[-1])
                if param_space in ("log"):
                    lb.append(np.log(param_range[0]))
                    ub.append(np.log(param_range[1]))
                elif param_space in ("logit"):
                    lb.append(logit(param_range[0]))
                    ub.append(logit(param_range[1]))
                else:
                    lb.append(param_range[0])
                    ub.append(param_range[-1])
            elif param_type == "bool":
                bb_input_type_string += 'B '
                #  bb_lower_bound_string += ' -'
                #  bb_upper_bound_string += ' -' # automatic bounds for boolean variables
                lb.append(0)
                ub.append(1)
            elif param_type in ("cat", "ordinal"):
                # the variables will be considered as integer periodic variables
                assert param_range is None
                bb_input_type_string += 'I '
                #  bb_lower_bound_string += ' 0'
                #  bb_upper_bound_string += ' ' + str(len(param_values) - 1)
                lb.append(0)
                ub.append(len(param_values) - 1)
                periodic_var_indexes.append(counter_periodic_vars)
            elif param_type == "real":
                # TODO: will we have to deal with with type of space (log, logit) ?
                bb_input_type_string += 'R '
                # Nomad has a tendance to go the bounds and the blackbox does not appreciate that
                if param_space in ("log"):
                    lb.append(np.log(param_range[0] + tol))
                    ub.append(np.log(param_range[1] - tol))
                elif param_space in ("logit"):
                    lb.append(logit(param_range[0] + tol))
                    ub.append(logit(param_range[1] - tol))
                else:
                    lb.append(param_range[0] + tol)
                    ub.append(param_range[-1] - tol)
                    #  bb_lower_bound_string += ' ' + str(param_range[0])
                    #  bb_upper_bound_string += ' ' + str(param_range[1])
            else:
                assert False, "type %s not handled in API" % param_type

            counter_periodic_vars += 1

        bb_input_type_string += ')'
        #  bb_lower_bound_string += ' )'
        #  bb_upper_bound_string += ' )'


        return bb_input_type_string, lb, ub, periodic_var_indexes, round_to_values #bb_input_type_string, bb_lower_bound_string, bb_upper_bound_string, periodic_var_indexes, round_to_values

    def bb_fct(self, x):
        try:
            n_values = x.get_n()

            dim_pb = len(self.api_config.keys())
            if ( n_values % dim_pb != 0 ):
                print("Invalid number of values passed to bb")
                return -1

            n_pts = n_values // dim_pb

            print("Number of points:", n_pts)

            # store the input points
            candidates = []
            for i in range(n_pts):
                candidates.append([x.get_coord(j) for j in range(i*dim_pb,(i+1)*dim_pb)])
            #  print("candidates")
            #  for candidate in candidates:
            #      print(candidate)
            self.inputs_queue.put(candidates)
            self.inputs_queue.join()

            # wait until the blackbox returns observations
            while self.outputs_queue.empty():
                continue

            # returns observations to the blackbox
            outputs_candidates = self.outputs_queue.get()
            for output_val in outputs_candidates:
                x.set_bb_output(i, output_val)

            # task finish
            self.outputs_queue.task_done()

        except:
            print ("Unexpected error in bb()", sys.exc_info()[0])
            return -1
        return 1


    def suggest(self, n_suggestions=1):

        """Make `n_suggestions` suggestions for what to evaluate next.
        This requires the user observe all previous suggestions before calling
        again.

        Parameters
        ----------
        n_suggestions : int
            The number of suggestions to return.
        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        # clear candidates before a new suggestion
        self.stored_candidates.clear()

        # wait until Nomad gives candidates
        while self.inputs_queue.empty():
            continue

        # collect candidates
        candidates = self.inputs_queue.get()

        assert len(candidates) >= 1, "No candidates given: error !"
        assert len(candidates) <= 8, "Too many candidates, n_suggestions must not be superior to 8 !"

        # put them in the framework model
        param_list = sorted(self.api_config.keys())

        next_guess = list()
        for candidate in candidates:
            guess = dict()

            for (param_name, val) in zip(param_list, candidate):

                param_config = self.api_config[param_name]
                param_type = param_config["type"]

                param_space = param_config.get("space", None)
                param_range = param_config.get("range", None)
                param_values = param_config.get("values", None)

                if param_type == "int":
                    if param_space in ("log"):
                        guess[param_name] = np.round(np.exp(val))
                    elif param_space in ("logit"):
                        guess[param_name] = np.round(expit(val))
                    else:
                        guess[param_name] = val
                elif param_type == "bool":
                    guess[param_name] = val
                elif param_type in ("cat", "ordinal"):
                    guess[param_name] = param_values[val]
                elif param_type == "real":
                    if param_space in ("log"):
                        guess[param_name] = np.exp(val)
                    elif param_space in ("logit"):
                        guess[param_name] = expit(val)
                    else:
                        guess[param_name] = val

                #  # make correspondance between periodic variables and categorical
                #  if param_type in ("cat", "ordinal"):
                #      guess[param_name] = param_values[val]
                #  else:
                #      guess[param_name] = val

            # round problematic variables
            for param_name, round_f in self.round_to_values.items():
                guess[param_name] = round_f(guess[param_name])

            # Also ensure this is correct dtype so sklearn is happy (according to hyperopt)
            guess = {k: DTYPE_MAP[self.api_config[k]["type"]](guess[k]) for k in guess}

            next_guess.append(guess)

            self.stored_candidates.append(guess)

        # complete task
        self.inputs_queue.task_done()

        # sometimes, the block is not filled: we have to complete it
        # In this case, via random points
        if 8 - len(candidates) > 0:
            random_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=8 - len(candidates), random=self.random)
            for guess in random_guess:
                next_guess.append(guess)

        return next_guess


    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y), "The length is not the same"

        outputs_candidates = list()

        # collect outputs
        for candidate in self.stored_candidates:
            idx = [candidate == x_guess for x_guess in X]
            id_y = np.argwhere(idx)[0].item() # pick the first index
            outputs_candidates.append(y[id_y])

        #  print(outputs_candidates)

        # trigger callbacks

        if self.nomad_process.is_alive():
        #  if self.nomad_thread.is_alive():
            self.outputs_queue.put(outputs_candidates)
            # wait for completion
            self.outputs_queue.join()
            self.n_iters += 1
            print("Observe Done!")

        # kill thread if last iteration
        if self.n_iters >= 16:
            self.nomad_process.terminate()
            self.nomad_process.join()


if __name__ == "__main__":
    experiment_main(PyNomadOptimizer)
