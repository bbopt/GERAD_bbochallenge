import numpy as np
from scipy.interpolate import interp1d
from PyNomad import optimize as nomad_solve

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

class PyNomadOptimizer(AbstractOptimizer):

    primary_import = "PyNomad"

    def __init__(self, api_config, n_initial_points=0):
        """Build wrapper class to use an optimizer in benchmark

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        n_initial_points : int
            Number of points to sample randomly before actual Nomad optimization
        """
        AbstractOptimizer.__init__(self, api_config)

        main_params = PyNomadOptimizer.get_nomad_dimensions(api_config)
        params = [main_params[0], main_params[1], main_params[2], 'BB_OUTPUT_TYPE OBJ', 'MAX_BB_EVAL ' + str(8 * 14)]
        x0 = [] # TODO choose x0 or LHS with n_initial_points




    @staticmethod
    def get_nomad_dimensions(api_config):
        """Help routine to setup PyNomad search space in constructor

            Take api_config in argument so this can be static
        """
        # there is equally a JointSpace method which could be used (see pysot)

        # take a look for argument; should be sorted normally
        # just in case we do it
        param_list = sorted(api_config.keys())

        bb_input_type_string = 'BB_INPUT_TYPE ('
        bb_lower_bound_string = 'LOWER_BOUND ('
        bb_upper_bound_string = 'UPPER_BOUND ('

        for param_name in param_list:
            param_config = api_config[param_name]
            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            #  # some setup for cardinality or ordinality variables
            #  # TODO : is there a special thing to do ? Do not understand this line
            #  values_only_type = param_type in ("cat", "ordinal")
            #  if (param_values is not None) and (not values_only_type):
            #      assert param_range is None
            #      param_values = np.unique(param_values)
            #      param_range = (param_values[0], param_values[-1])
            #      round_to_values[param_name] = interp1d(
            #          param_values, param_values, kind="nearest", fill_value="extrapolate"
            #      )

            if param_type == "int":
                bb_input_type_string += 'I '
                bb_lower_bound_string += ' ' + str(param_range[0])
                bb_upper_bound_string += ' ' + str(param_range[-1])
            elif param_type == "bool":
                # TODO: how nomad deals with boolean variables ? Will have to change lower and upper bounds values
                bb_input_type_string += 'B '
                bb_lower_bound_string += ' 0'
                bb_upper_bound_string += ' 1'
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                bb_input_type_string += 'C '
                bb_lower_bound_string += ' -' # TODO at this time no upper bound neither lower bound; must be put
                bb_upper_bound_string += ' -'
            elif param_type == "real":
                # TODO: will we have to deal with with type of space (log, logit) ?
                bb_input_type_string += 'R '
                bb_lower_bound_string += ' ' + str(param_range[0])
                bb_upper_bound_string += ' ' + str(param_range[1])
            else:
                assert False, "type %s not handled in API" % param_type

        bb_input_type_string += ')'
        bb_lower_bound_string += ')'
        bb_upper_bound_string += ')'

        return bb_input_type_string, bb_lower_bound_string, bb_upper_bound_string

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










