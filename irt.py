class RaschModelEstimation:
    """
    Calculate expected ability > this provides maximum likelihood
    """

    def __init__(self, *student_info, **lst_of_difficulty):
        self.student_info = student_info
        self.lst_of_difficulty = lst_of_difficulty

    def name(self):
        for name in self.student_info:
            if type(name) == str:
                return name

    def probability_of_success(self):
        from scipy.special import expit
        """
        expit(x) = 1/(1+exp(-x))
        """

        for theta in self.student_info:
            if type(theta) == int or type(theta) == float:
                for diff in self.lst_of_difficulty["lst_of_difficulty"]:
                    # yield float(expit(theta - diff))
                    yield round(float(expit(theta - diff)), 2)

    @staticmethod
    def lst_variance(lst_probability_success, lst_probability_failure):
        if len(lst_probability_success) == 0 or len(lst_probability_failure) == 0:
            raise ValueError("Need lists of success  es and failures!")
        return [success * failure for success, failure in zip(lst_probability_success, lst_probability_failure)]

    @staticmethod
    def newton_raphson(expected_ability=None, observed_score=None, expected_score=None, total_variance=None):
        """
        Using newton_raphson model
        0 : z_score > (obs - exp)/ total variance
        1 : Newton-Raphson method
        """
        if expected_ability is None or observed_score is None or expected_ability is None or total_variance is None:
            raise ValueError("Unknown variables!")
        return round((observed_score - expected_score) / total_variance, 2), \
               round(expected_ability + (observed_score - expected_score) / total_variance, 2)


def rasch_analysis(name, ability, ability_hat, difficulty, item_result):
    import numpy as np
    print("\n  << Rasch Measurement >>  \n  << probability_of_success >>\n")
    no = 0
    while ability_hat < 100:
        print("  Iteration {}".format(no))
        no += 1
        expected_ability = RaschModelEstimation(name, ability, lst_of_difficulty=difficulty)
        lst_success = list(expected_ability.probability_of_success())
        lst_failure = [one-success for one, success in zip([1] * len(lst_success), lst_success)]
        lst_variance = expected_ability.lst_variance(lst_success, lst_failure)
        total_var = sum(lst_variance)

        score_hat = sum(lst_success)
        obs_score = sum(item_result)

        print("  {}'s ability: {} (ability_hat: {})\n  difficulty: {}\n  expected success: {}  = {}\n  expected failures: {}\n  variance (statistical information): {}\n  total variance: {}".format(
            expected_ability.name(), ability, ability_hat, difficulty, lst_success, sum(lst_success), lst_failure, lst_variance, round(total_var, 2)))

        ability_hat = expected_ability.newton_raphson(observed_score=obs_score, expected_score=score_hat,
                                              expected_ability=ability, total_variance=total_var)[0]
        updated_ability = expected_ability.newton_raphson(observed_score=obs_score, expected_score=score_hat,
                                              expected_ability=ability, total_variance=total_var)[1]
        print("  ability_hat: {}, updated_ability: {}\n".format(ability_hat, updated_ability))

        ability += ability_hat

        if ability_hat == 0:
            print("  >> {}".format(name))
            print("  >> Estimated ability: {}".format(updated_ability))
            print("  >> item_result: {}".format(item_result))

            print("\n  >> Maximum Likelihood")
            t_lst = sorted(list(set(difficulty))) + [updated_ability]
            t_lst.sort()
            for ability_h in t_lst:
                expected_ability = RaschModelEstimation(name, ability_h, lst_of_difficulty=difficulty)
                lst_success = list(expected_ability.probability_of_success())
                lst_failure = [one - success for one, success in zip([1] * len(lst_success), lst_success)]
                expected_success = [a*b for a, b in zip(lst_success, item_result)]
                expected_failure = [a*b for a, b in
                                    zip(lst_failure, [abs(a-b) for a, b in zip(item_result, [1] * len(item_result))])]
                item_result_prob = [a+b for a, b in zip(expected_success, expected_failure)]
                if len(str(ability_h)) == 1:
                    # print("  >> list of probabilities: {}".format(item_result_prob))
                    print("  >> Estimated ability: {}.00 Probability: {}".format(ability_h, np.prod(item_result_prob)))
                else:
                    # print("  >> list of probabilities: {}".format(item_result_prob))
                    print("  >> Estimated ability: {} Probability: {}".format(ability_h, np.prod(item_result_prob)))
            # fit statistics
            outfit = round(sum([np.square(a - b) / c for a, b, c in zip(item_result, lst_success, lst_variance)]) / len(item_result), 2)
            infit = round(sum([np.square(a - b) for a, b in zip(item_result, lst_success)]) / sum(lst_variance), 2)
            print("\n  >> Outfit: {}\n  >> Infit: {}\n ".format(outfit, infit))
            break
    return ability

"""
example )
name = 'unique_user'
ability = 3
ability_hat = 0
difficulty = [1, 2, 3, 4, 1, 2, 3, 3, 2, 3, 1, 2, 3, 4, 4, 4, 3, 4, 5, 5]
item_result = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]

student_ability = rasch_analysis(test_takers, theta, theta_hat, lst_difficulty, result)
print(student_ability)
"""


class IrtModel:

    def __init__(self, theta, a, b, c):
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c

    def logit(self):
        return round(self.a * (self.theta - self.b), 3)

    def exp_neg_logit(self):
        import math

        return round(math.exp(-self.a * (self.theta - self.b)), 3)

    def three_parameter_logistic_model(self):
        from scipy.special import expit

        return round(self.c + (1 - self.c) * expit(self.a * (self.theta - self.b)), 2)

    def plot_item_characteristic_curve(self, x_data, y_data, label, title, xlabel, ylabel):
        import matplotlib.pyplot as plt
        from scipy.interpolate import make_interp_spline
        import numpy as np

        plt.scatter(x_data, y_data, c='b', marker='x', label='{}'.format(label)) # person A
        xnew = np.linspace(min(x_data), max(x_data), 300)
        spl = make_interp_spline(x_data, y_data, k=3)
        power_smooth = spl(xnew)
        plt.plot(xnew, power_smooth)
        plt.legend(loc='upper right')
        plt.title('{}'.format(title)) # Maximum Likelihood
        plt.xlabel('{}'.format(xlabel)) # Expected Ability
        plt.ylabel('{}'.format(ylabel)) # Probability
        plt.ylim(ymin=0, ymax=1)
        plt.show()


class InternalConsistency:
    """
    col_user = string
    col_total_number_of_questions = list
    """

    def __init__(self, dataframe, col_user, col_total_number_of_questions):
        self.dataframe = dataframe
        self.col_user = col_user # "unique user"
        self.col_total_number_of_questions = col_total_number_of_questions # "number_of_questions"

    def cronbachs_alpha(self):
        import pandas as pd

        df = pd.DataFrame()

        self.dataframe[self.col_total_number_of_questions] = self.dataframe["result"].str.len()
        if self.dataframe[self.col_total_number_of_questions].nunique() != 1:
            raise ValueError("Need to check data..")

        for ind in self.dataframe.index:
            df_append = pd.DataFrame({self.col_user: [self.dataframe.loc[ind, self.col_user]]})

            for r in range(self.dataframe.loc[ind, self.col_total_number_of_questions]):
                df_append["item_{}".format(r + 1)] = self.dataframe.loc[ind, "result"][r]
                del r

            df_append = df_append.set_index([self.col_user])

            df = df.append(df_append, sort=False)
            del ind, df_append

        lst_test_score = []
        for ind in df.index:
            lst_test_score.append(df.loc[ind, df.columns].sum())
            del ind

        lst_item_variance = []
        for col in df.columns:
            lst_item_variance.append(df[col].var())

        number_of_item = len(lst_item_variance)
        item_variance = sum(lst_item_variance)
        test_score_variance = pd.DataFrame({"total_score": lst_test_score})
        test_score_variance = test_score_variance["total_score"].var()

        cronbachs_value = (number_of_item / (number_of_item - 1)) * (
                (test_score_variance - item_variance) / test_score_variance)

        return cronbachs_value
