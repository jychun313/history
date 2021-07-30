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
