import logging

import numpy as np

from scipy.spatial.distance import euclidean


class WeightEstimator:

    # default constructor
    def __init__(self, delta: float = 0.1, n_objectives: int = 2, initial_weights: tuple = None):
        assert n_objectives >= 2
        # Set logger with class self name
        self.log = logging.getLogger(self.__class__.__name__)
        self.visited_pairs = []
        # set delta
        self.delta = delta
        self.weight_candidates = []
        self.optimal_results = []
        # dict using weights as keys and function results as values. The key is converted from nd.array to tuple because
        # nd.arrays are not hashable, probably because they are mutable
        self.results = {}
        self.weight_index = -1
        # init weights
        if initial_weights is not None:
            [self.weight_candidates.append(weight) for weight in initial_weights]
        else:
            for i in range(n_objectives):
                w = np.zeros([n_objectives])
                w[i] = 1
                self.weight_candidates.append(w)

    def has_next(self) -> bool:
        # check if all candidates were visited
        if self.weight_index == len(self.weight_candidates) - 1:
            # check if we can calculate more or not
            if self.calculate_next() <= self.delta:
                return False
        return True

    def calculate_next(self) -> float:
        opt1, opt2 = self.find_optimal_pair()
        distance = 0
        if opt1 is not None and opt2 is not None:
            weight = self.calculate_next_weight(opt1, opt2)
            self.log.info("w = %s, opt1 = %s, opt2 = %s", weight, opt1, opt2)
            self.weight_candidates.append(weight)
            distance = self.euclidean_distance(opt1, opt2)
            self.log.info("distance = %s", distance)
        return distance

    def find_optimal_pair(self) -> list:
        last_opt = None
        max_distance = 0
        pair = [None, None]
        for curr_opt in self.optimal_results:
            if last_opt is None:
                last_opt = curr_opt
                continue
            # skip pairs that were already visited
            if self.was_visited(last_opt, curr_opt):
                continue
            distance = self.euclidean_distance(last_opt, curr_opt)
            if distance > max_distance:
                max_distance = distance
                pair = [last_opt, curr_opt]
            last_opt = curr_opt

        # mark as visited
        self.visited_pairs.append(np.vstack(pair))
        return pair

    @staticmethod
    def calculate_next_weight(opt1: np.ndarray, opt2: np.ndarray) -> np.ndarray:
        """
        This method resolve a simple linear equation. It finds the weights (line) that connects
        the two parameters opt1 and opt2 with the constraint to have an result with sum 1.
        :param opt1:    an array with the value for each dimension
        :param opt2:    a second array with the value for each dimension
        :return:        the solution w for the equation aw = b
        """
        assert opt1.shape == opt2.shape, "Both arrays should have the same size, but {} and {} were given".format(
            opt1.shape, opt2.shape)
        # for 2D case it would be:
        # w1 * (opt2[0] - opt1[0]) + w2 * (op2[1] - op1[1]) = 0
        # w1 * 1                   + w2 * 1                 = 1
        multipliers = np.array([opt2 - opt1, np.ones(opt1.shape)])
        result = np.array([0, 1])
        # solve the linear equation
        return np.linalg.solve(multipliers, result)

    def set_result(self, result: np.ndarray, weight: np.ndarray):
        assert len(weight) == len(result)
        self.results[tuple(weight)] = result
        # if it is not dominated add it to optimal_results
        if not self.is_dominated(result):
            # calculate the index to insert
            reached_end = True
            index = 0
            for i, opt in enumerate(self.optimal_results):
                if opt[1] < result[1]:
                    index = i
                    reached_end = False
                    break
            if reached_end:
                self.optimal_results.append(result)
            else:
                # insert where it stops to be bigger
                self.optimal_results.insert(index, result)
        # remove those optimal that result dominates
        # TODO we may not want to remove the last item, it may be a good approach to let it here,
        # although it would generate negative values of w
        self.optimal_results = [optimal for optimal in self.optimal_results if np.any(np.less(optimal, result))
                                or np.array_equal(optimal, result)]

    def is_dominated(self, result: np.ndarray) -> bool:
        """
        This method check if one result is dominated by the current optimal results.
        This method expected smaller values to be better, so a result is dominated
        if it is bigger than some optimal results in all objectives.
        :param result:  the array with objective results values
        :return:        True if the result is dominated by some other optimal results,
                        False otherwise
        """
        for opt_objective in self.optimal_results:
            # if the result is greater in all objectives, then it is dominated
            if np.all(np.greater_equal(result, opt_objective)):
                return True
        return False

    def get_next(self) -> np.ndarray:
        self.weight_index += 1
        return self.weight_candidates[self.weight_index]

    def was_visited(self, last_obj: np.ndarray, curr_obj: np.ndarray):
        curr_pair = np.vstack([last_obj, curr_obj])
        for pair in self.visited_pairs:
            if np.array_equal(pair, curr_pair):
                return True
        return False

    def euclidean_distance(self, last_opt, curr_opt):
        # TODO improve it for negative values and min != 0
        # max value per objective array
        max_w = np.max(np.array(self.optimal_results), axis=0)
        # divide all by max_w to normalize the max value to 1
        return euclidean(last_opt / max_w, curr_opt / max_w)
