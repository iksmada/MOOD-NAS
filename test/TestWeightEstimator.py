from unittest import TestCase, SkipTest

import numpy as np
import matplotlib.pyplot as plt

import train_search
from train_search import create_parser, L2_LOSS, CRITERION_LOSS, REG_LOSS
from WeightEstimator import WeightEstimator


class TestWeightEstimator(TestCase):

    def setUp(self) -> None:
        self.instance = WeightEstimator()
        self.sqr_fun = lambda x: (x - 1) ** 2
        self.linear_fun = lambda x: (x - 1)

    def consume_w_and_set(self, fun=lambda x: (x - 1) ** 2):
        weight = self.instance.get_next()
        result = fun(weight)
        self.instance.set_result(result, weight)
        return weight

    def test_init(self):
        np.testing.assert_array_equal(self.instance.get_next(), np.array([1, 0]))
        np.testing.assert_array_equal(self.instance.get_next(), np.array([0, 1]))

    def test_set_result(self):
        for i in range(2):
            weight = self.consume_w_and_set()
            self.assertEquals(len(self.instance.optimal_results), i + 1)
            # check if the result from that weight is optimal
            self.assertTrue((self.instance.results[tuple(weight)] == self.instance.optimal_results).any())
            # check if the weight is on the list
            self.assertIn(tuple(weight), self.instance.results.keys())

    def test_is_dominated(self):
        self.consume_w_and_set()
        self.consume_w_and_set()
        self.assertTrue(self.instance.is_dominated(np.array([1, 1])))
        self.assertFalse(self.instance.is_dominated(np.array([0, 0])))

    def test_calculate_next_weight_linear(self):
        self.consume_w_and_set(self.linear_fun)
        self.consume_w_and_set(self.linear_fun)
        results = list(self.instance.optimal_results)
        opt1 = results[0]
        opt2 = results[1]
        np.testing.assert_array_equal(WeightEstimator.calculate_next_weight(opt1, opt2), np.array([1 / 2, 1 / 2]))

    def test_calculate_next_weight(self):
        opt1 = np.array([1, 1 / 3])
        opt2 = np.array([0, 1])
        np.testing.assert_array_equal(WeightEstimator.calculate_next_weight(opt1, opt2), np.array([2 / 5, 3 / 5]))
        opt1 = np.array([1, 0])
        opt2 = np.array([0, 0])
        np.testing.assert_array_equal(WeightEstimator.calculate_next_weight(opt1, opt2), np.array([0, 1]))

    def test_find_optimal_pair(self):
        w1 = np.array([0, 1])
        result1 = np.array([0, 1])
        self.instance.set_result(result1, w1)
        w2 = np.array([1, 0])
        result2 = np.array([1, 0])
        self.instance.set_result(result2, w2)
        # generate one more w
        self.instance.has_next()
        w3 = self.instance.get_next()
        result3 = np.array([0.1, 0.9])
        self.instance.set_result(result3, w3)
        pair = self.instance.find_optimal_pair()
        # the order of neighbours is result1, result3, result2 because they are ordered by the frontier
        np.testing.assert_array_equal(result3, pair[0])
        np.testing.assert_array_equal(result2, pair[1])

    def test_has_next(self):
        while self.instance.has_next():
            self.consume_w_and_set(self.sqr_fun)
            print(self.instance.weight_candidates[-1])
            # assert that this ends in less than 50 iterations
            max_iter = 50
            self.assertLess(len(self.instance.weight_candidates), max_iter,
                            "This should end in less than {} iterations".format(max_iter))
        plt.scatter(list(pair[0] for pair in self.instance.optimal_results),
                    list(pair[1] for pair in self.instance.optimal_results))
        plt.show()

    def test_random_function(self):
        # this test is failing now because we are ordering the frontier according to y values
        # this was working fine when we were saving the index to where to save on optimal_results
        while self.instance.has_next():
            random_fun = lambda x: self.sqr_fun(x) + (np.random.random(2) - 0.5) / 30
            self.consume_w_and_set(random_fun)
            print(self.instance.weight_candidates[-1])
            # assert that this ends in less than 200 iterations
            max_iter = 200
            self.assertLess(len(self.instance.weight_candidates), max_iter,
                            "This should end in less than {} iterations".format(max_iter))
        plt.scatter(list(pair[0] for pair in self.instance.results.values()),
                    list(pair[1] for pair in self.instance.results.values()),
                    label="Dominated Results")
        plt.scatter(list(pair[0] for pair in self.instance.optimal_results),
                    list(pair[1] for pair in self.instance.optimal_results),
                    label="Optimal Results")
        plt.legend()
        plt.show()

    @SkipTest
    def test_real_network(self):
        while self.instance.has_next():
            parser = create_parser()
            args = parser.parse_args()
            args.layers = 1
            weight = self.instance.get_next()
            args.l2_weight = weight[0]
            args.criterion_weight = weight[1]
            args.l1_weight = 0
            stats = train_search.main(args)
            reg_loss = stats.get(L2_LOSS).get(weight).get(REG_LOSS)
            criterion_loss = stats.get(L2_LOSS).get(weight).get(CRITERION_LOSS)
            self.instance.set_result(np.array[reg_loss, criterion_loss], weight)
            self.assertLess(len(self.instance.weight_candidates), 50, "This should end in less than 50 iterations")

        plt.scatter(list(pair[0] for pair in self.instance.optimal_results),
                    list(pair[1] for pair in self.instance.optimal_results))
        plt.show()



