"""
Initial LH search: 16 points
Score (local): 92
Score (distant): 82.288
"""
import numpy as np
from scipy.interpolate import interp1d
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
        params = [main_params[0], #main_params[1], main_params[2],
                  'BB_OUTPUT_TYPE OBJ',
                  'MAX_BB_EVAL ' + str(8 * 16),
                  'BB_MAX_BLOCK_SIZE 8',
                  'MODEL_SEARCH SGTELIB',
                  'SGTELIB_MODEL_CANDIDATES_NB 8',
                  'SGTELIB_MODEL_TRIALS 5',
                  'MODEL_EVAL_SORT no',
                  'DIRECTION_TYPE ORTHO 2N',
                  'SPECULATIVE_SEARCH no',
                  'LH_SEARCH 16 0',
                  'OPPORTUNISTIC_EVAL false',
                  'NM_SEARCH false'] #,'PERIODIC_VARIABLE 0'] 

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

                # make correspondance between periodic variables and categorical
                if param_type in ("cat", "ordinal"):
                    guess[param_name] = param_values[val]
                else:
                    guess[param_name] = val

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
