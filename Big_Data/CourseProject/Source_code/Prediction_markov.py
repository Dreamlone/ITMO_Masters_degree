import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import gdal, osr
import scipy.stats
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from pomegranate import *
import itertools

from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams


class MarkovBlanket():
    """
    An object for storing info on nodes within the markov blanket of the hidden node

    Parameters
    ----------
    ind_h : int
        index of the hidden node within the model

    Attributes
    ----------
    hidden : int
        index of the hidden node

    parents : list of int
        a list of indices of the parent nodes

    children : list of int
        a list of indices of the children nodes

    coparents : list of int
        a list of indices of the coparent nodes

    prob_table : dict
        a dict of probabilities table of nodes within the Markov blanket

    """

    def __init__(self, ind_h):
        self.hidden = ind_h
        self.parents = []
        self.children = []
        self.coparents = []
        self.prob_table = {}

    def populate(self, model):
        """populate the parents, children, and coparents nodes
        """
        state_indices = {state.name: i for i, state in enumerate(model.states)}

        edges_list = [(parent.name, child.name) for parent, child in model.edges]
        edges_list = [(state_indices[parent], state_indices[child])
                      for parent, child in edges_list]

        self.children = list(set([child for parent, child in edges_list if parent == self.hidden]))
        self.parents = list(set([parent for parent, child in edges_list if child == self.hidden]))
        self.coparents = list(set([parent for parent, child in edges_list if child in self.children]))
        try:
            self.coparents.remove(self.hidden)
        except ValueError:
            pass

    def calculate_prob(self, model):
        """Create the probability table from nodes
        """
        for ind_state in [self.hidden] + self.children:
            distribution = model.states[ind_state].distribution

            if isinstance(distribution, ConditionalProbabilityTable):
                table = list(distribution.parameters[0])  # make a copy
                self.prob_table[ind_state] = {
                    tuple(row[:-1]): row[-1] for row in table}
            else:
                self.prob_table[ind_state] = dict(distribution.parameters[0])  # make a copy

    def update_prob(self, model, expected_counts, ct):
        """Update the probability table using expected counts
        """
        ind = {x: i for i, x in enumerate([self.hidden] + self.parents + self.children + self.coparents)}
        mb_keys = expected_counts.counts.keys()

        for ind_state in [self.hidden] + self.children:
            distribution = model.states[ind_state].distribution

            if isinstance(distribution, ConditionalProbabilityTable):
                idxs = distribution.column_idxs
                table = self.prob_table[ind_state]  # dict

                # calculate the new parameter for this key
                for key in table.keys():
                    num = 0
                    denom = 0

                    # marginal counts
                    for mb_key in mb_keys:
                        # marginal counts of node + parents
                        if tuple([mb_key[ind[x]] for x in idxs]) == key:
                            num += ct.table[mb_key[1:]] * expected_counts.counts[mb_key]

                            # marginal counts of parents
                        if tuple([mb_key[ind[x]] for x in idxs[:-1]]) == key[:-1]:
                            denom += ct.table[mb_key[1:]] * expected_counts.counts[mb_key]

                    try:
                        prob = num / denom
                    except ZeroDivisionError:
                        prob = 0

                    # update the parameter
                    table[key] = prob

            else:  # DiscreteProb
                table = self.prob_table[ind_state]  # dict

                # calculate the new parameter for this key
                for key in table.keys():
                    prob = 0
                    for mb_key in mb_keys:
                        if mb_key[ind[ind_state]] == key:
                            prob += ct.table[mb_key[1:]] * expected_counts.counts[mb_key]

                    # update the parameter
                    table[key] = prob / ct.size

class ExpectedCounts():
    """Calculate the expected counts using the model parameters

    Parameters
    ----------
    model : a BayesianNetwork object

    mb : a MarkovBlanket object

    Attributes
    ----------
    counts : dict
        a dict of expected counts for nodes in the Markov blanket
    """

    def __init__(self, model, mb):
        self.counts = {}

        self.populate(model, mb)

    def populate(self, model, mb):
        # create combinations of keys
        keys_list = [model.states[mb.hidden].distribution.keys()]
        for ind in mb.parents + mb.children + mb.coparents:
            keys_list.append(model.states[ind].distribution.keys())

        self.counts = {p: 0 for p in itertools.product(*keys_list)}

    def update(self, model, mb):
        ind = {x: i for i, x in enumerate([mb.hidden] + mb.parents + mb.children + mb.coparents)}

        marginal_prob = {}

        # calculate joint probability and marginal probability
        for i, key in enumerate(self.counts.keys()):
            prob = 1

            for j, ind_state in enumerate([mb.hidden] + mb.children):
                distribution = model.states[ind_state].distribution

                if isinstance(distribution, ConditionalProbabilityTable):
                    idxs = distribution.column_idxs
                    state_key = tuple([key[ind[x]] for x in idxs])
                else:
                    state_key = key[ind[ind_state]]

                prob = prob * mb.prob_table[ind_state][state_key]
                self.counts[key] = prob
            try:
                marginal_prob[key[1:]] += prob
            except KeyError:
                marginal_prob[key[1:]] = prob

        # divide the joint prob by the marginal prob to get the conditional
        for i, key in enumerate(self.counts.keys()):
            try:
                self.counts[key] = self.counts[key] / marginal_prob[key[1:]]
            except ZeroDivisionError:
                self.counts[key] = 0


class CountTable():
    """Counting the data"""

    def __init__(self, model, mb, items):
        """
        Parameters
        ----------
        model : BayesianNetwork object

        mb : MarkovBlanket object

        items : ndarray
            columns are data for parents, children, coparents

        """
        self.table = {}
        self.ind = {}
        self.size = items.shape[0]

        self.populate(model, mb, items)

    def populate(self, model, mb, items):
        keys_list = []
        for ind in mb.parents + mb.children + mb.coparents:
            keys_list.append(model.states[ind].distribution.keys())

        # init
        self.table = {p: 0 for p in itertools.product(*keys_list)}
        self.ind = {p: [] for p in itertools.product(*keys_list)}

        # count
        for i, row in enumerate(items):
            try:
                self.table[tuple(row)] += 1
                self.ind[tuple(row)].append(i)
            except KeyError:
                print('Items in row', i, 'does not match the set of keys.')
                raise KeyError


def em_bayesnet(model, data, ind_h, max_iter=50, criteria=0.005):
    """Returns the data array with the hidden node filled in.
    (model is not modified.)

    Parameters
    ----------
    model : a BayesianNetwork object
        an already baked BayesianNetwork object with initialized parameters

    data : an ndarray
        each column is the data for the node in the same order as the nodes in the model
        the hidden node should be a column of NaNs

    ind_h : int
        index of the hidden node

    max_iter : int
        maximum number of iterations

    criteria : float between 0 and 1
        the change in probability in consecutive iterations, below this value counts as convergence

    Returns
    -------
    data : an ndarray
        the same data arary with the hidden node column filled in
    """

    # create the Markov blanket object for the hidden node
    mb = MarkovBlanket(ind_h)
    mb.populate(model)
    mb.calculate_prob(model)

    # create the count table from data
    items = data[:, mb.parents + mb.children + mb.coparents]
    ct = CountTable(model, mb, items)

    # create expected counts
    expected_counts = ExpectedCounts(model, mb)
    expected_counts.update(model, mb)

    # ---- iterate over the E-M steps
    i = 0
    previous_params = list(mb.prob_table[mb.hidden].values())
    convergence = False

    while (not convergence) and (i < max_iter):
        mb.update_prob(model, expected_counts, ct)
        expected_counts.update(model, mb)
        # print 'Iteration',i,mb.prob_table

        # convergence criteria
        hidden_params = list(mb.prob_table[mb.hidden].values())
        change = np.abs([hidden_params[0] - previous_params[0], hidden_params[1] - previous_params[1]])
        convergence = max(change) < criteria
        previous_params = list(mb.prob_table[mb.hidden].values())

        i += 1

    if i == max_iter:
        print('Maximum iterations reached.')

    # ---- fill in the hidden node data by sampling the distribution
    labels = {}
    for key, prob in expected_counts.counts.items():
        try:
            labels[key[1:]].append((key[0], prob))
        except:
            labels[key[1:]] = [(key[0], prob)]

    for key, counts in ct.table.items():
        label, prob = zip(*labels[key])
        prob = tuple(round(p, 5) for p in prob)
        if not all(p == 0 for p in prob):
            samples = np.random.choice(label, size=counts, p=prob)
            data[ct.ind[key], ind_h] = samples

    return data


#########################################################################################################################
# =============================         Методы, необходимые для применения модели         ============================= #
#########################################################################################################################


# Функция предсказания для выбранного года значения урожайности
def predict(path, year, country, crop, random_val):

    # Доступ к конкретному файлу с датасетом
    files = os.listdir(path)
    for file in files:
        if file.startswith(country):
            file_path = os.path.join(path, file)
    dataframe = pd.read_csv(file_path)
    dataframe['year'] = dataframe['Year']
    dataframe.set_index('Year', inplace=True)

    # Оставляем только те данные, которые имелись на момент предсказания
    new_dataframe = dataframe.loc[1980:year] # Год, для которого делается предсказание, еще есть в данном датасете

    # Последовательность действий:
    # Урожайность. Кластеризация
    # Теплообеспеченность. Кластеризация
    # Количество осадков. Кластеризация
    # Распределение поля давления. Кластеризация
    for parameter in [crop, 'Temperature_SAT', 'Precip_amount', 'Pressure_mean']:
        extr_parameter = np.array(new_dataframe[parameter])
        extr_parameter = extr_parameter.reshape((-1, 1))

        # Процедура кластеризации
        gmm = GaussianMixture(n_components = 3, random_state = random_val)
        gmm.fit(extr_parameter)
        clusters = gmm.predict(extr_parameter)
        clusters = np.array(clusters)

        # Полученные коды кластеров заносим в датафрейм
        name = parameter + '_cod'
        new_dataframe[name] = clusters
        gmm = None

    ###################################################################################################################
    #                                              Отрисовка кластеризации                                            #
    ###################################################################################################################
    rcParams['figure.figsize'] = 11, 6
    plot_crop_code = crop + '_cod'
    plot_new_dataframe_0 = new_dataframe[new_dataframe[plot_crop_code] == 0]
    plot_new_dataframe_1 = new_dataframe[new_dataframe[plot_crop_code] == 1]
    plot_new_dataframe_2 = new_dataframe[new_dataframe[plot_crop_code] == 2]

    plt.plot(new_dataframe['year'], new_dataframe[crop], c = 'blue', linewidth = 1, alpha = 0.2)
    plt.scatter(plot_new_dataframe_0['year'], plot_new_dataframe_0[crop], c = 'red', s = 50)
    plt.scatter(plot_new_dataframe_1['year'], plot_new_dataframe_1[crop], c = 'blue', s = 50)
    plt.scatter(plot_new_dataframe_2['year'], plot_new_dataframe_2[crop], c = 'green', s = 50)
    plt.ylabel(crop, fontsize=13)
    plt.xlabel('Year', fontsize=13)
    plt.title('Clusters', fontsize=13)
    plt.grid()
    plt.close()

    plot_parameter = 'Temperature_SAT'
    plot_crop_code = plot_parameter + '_cod'
    plot_new_dataframe_0 = new_dataframe[new_dataframe[plot_crop_code] == 0]
    plot_new_dataframe_1 = new_dataframe[new_dataframe[plot_crop_code] == 1]
    plot_new_dataframe_2 = new_dataframe[new_dataframe[plot_crop_code] == 2]

    plt.plot(new_dataframe['year'], new_dataframe[plot_parameter], c='blue', linewidth=1, alpha=0.2)
    plt.scatter(plot_new_dataframe_0['year'], plot_new_dataframe_0[plot_parameter], c='red', s=50)
    plt.scatter(plot_new_dataframe_1['year'], plot_new_dataframe_1[plot_parameter], c='blue', s=50)
    plt.scatter(plot_new_dataframe_2['year'], plot_new_dataframe_2[plot_parameter], c='green', s=50)
    # Sum of active temperatures
    plt.ylabel('Sum of active temperatures', fontsize=13)
    plt.xlabel('Year', fontsize=13)
    plt.title('Clusters', fontsize=13)
    plt.grid()
    plt.close()

    # Действительное значение урожайности для нужного нам года
    test_dataframe = new_dataframe.loc[year]
    actual = test_dataframe[crop]

    # По обучающей выборке составляем неободимые матрицы частот
    # DiscreteDistribution - для переменной отклика
    # ConditionalProbabilityTable - для предикторов
    all_rows = len(new_dataframe) # Всего количество строк в датафрейме

    # Насколько часто встречались в датафрейме те или иные значения урожайности
    crop_code = crop + '_cod'
    freq_dict = {}
    for number in range(0,3):
        frequency = len(new_dataframe[new_dataframe[crop_code] == number])/all_rows
        freq_dict.update({str(number): frequency})
    # Первое распределение частот готово
    CropYield = DiscreteDistribution(freq_dict)

    # Цикл задания таблицы ConditionalProbabilityTable для теплообеспеченности
    temperature = []
    rainfall = []
    pressure = []
    for number in range(0,3):
        # Рассматриваем конкретное состояние системы по урожайности
        local_data = new_dataframe[new_dataframe[crop_code] == number]
        # Сколько всего значений в датасете имеет значение number по урожайности
        local_all_rows = len(local_data)

        for j in range(0,3):
            # Сумма активных температур
            j_rows_TMP = len(local_data[local_data['Temperature_SAT_cod'] == j])
            j_frequency_TMP = j_rows_TMP/local_all_rows

            # Сумма осадков
            j_rows_PRC = len(local_data[local_data['Precip_amount_cod'] == j])
            j_frequency_PRC = j_rows_PRC/local_all_rows

            # Среднее давление
            j_rows_PRE = len(local_data[local_data['Pressure_mean_cod'] == j])
            j_frequency_PRE = j_rows_PRE/local_all_rows

            # Составляем массивы
            temperature.append([str(number), str(j), j_frequency_TMP])
            rainfall.append([str(number), str(j), j_frequency_PRC])
            pressure.append([str(number), str(j), j_frequency_PRE])

    # Массивы готовы для подачи в модель
    TemperatureSum = ConditionalProbabilityTable(temperature, [CropYield])
    RainfallSum = ConditionalProbabilityTable(rainfall, [CropYield])
    PressureMean = ConditionalProbabilityTable(pressure, [CropYield])

    s_yield = State(CropYield, 'yield')
    s_SAT = State(TemperatureSum, 'SAT')
    s_rainfall = State(RainfallSum, 'rainfall')
    s_pressure = State(PressureMean, 'pressure')
    model = BayesianNetwork('yield')

    model.add_states(s_yield, s_SAT, s_rainfall, s_pressure)
    model.add_transition(s_yield, s_SAT)
    model.add_transition(s_yield, s_rainfall)
    model.add_transition(s_yield, s_pressure)
    model.bake()

    # Данные для подачи в модель
    this_year = new_dataframe.loc[year]
    arr = np.array(this_year[[crop_code, 'Temperature_SAT_cod', 'Precip_amount_cod', 'Pressure_mean_cod']])
    data_to_model = [np.nan]
    for element in arr:
        data_to_model.append(str(element))
    data_to_model = np.array([data_to_model])

    #                                          --- Применение модели ---
    hidden_node_index = 0
    new_data = em_bayesnet(model, data_to_model, hidden_node_index)

    predicted_cluster = new_data[0][0]
    Data = new_dataframe.loc[:year-1]
    Data = Data[Data[crop_code] == int(predicted_cluster)]
    crop_array = np.array(Data[crop])
    mean_yield = np.mean(crop_array)

    print('Предсказанное значение кластера', predicted_cluster)
    print('Среднее значение урожайности по данному кластеру -', mean_yield)
    print('Действительно наблюдаемое значение -', actual, '\n')

    return(mean_yield, actual)

# Графики остатков + информация по метрикам
# Расчет метрики - cредняя абсолютная процентная ошибка
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # У представленной ниже формулы есть недостаток, - если в массиве y_true есть хотя бы одно значение 0.0,
    # то по формуле np.mean(np.abs((y_true - y_pred) / y_true)) * 100 мы получаем inf, поэтому
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return (value)


# Зададим функцию, которая будет выводить на экран значения метрик
def print_metrics(y_test, prediction):
    print('Mean absolute error in the test sample -', round(mean_absolute_error(y_test, prediction),2))
    print('Median absolute error in the test sample -', round(median_absolute_error(y_test, prediction),2))
    print('Root mean square error in the test sample -', round(mean_squared_error(y_test, prediction) ** 0.5, 2))
    print('Mean absolute percentage error in the test sample -', round(mean_absolute_percentage_error(y_test, prediction), 2))


# Зададим функцию для отрисовки графиков
def residuals_plots(y_test, prediction, color='blue'):
    prediction = np.ravel(prediction)
    y_test = np.ravel(y_test)
    # Рассчитываем ошибки
    errors = prediction - y_test
    errors = np.ravel(errors)

    plot_data = pd.DataFrame({'Errors': errors,
                              'Prediction': prediction})

    with sns.axes_style("ticks"):
        g = (sns.jointplot('Prediction', 'Errors', height=7, alpha=0.6,
                           data=plot_data, color=color).plot_joint(sns.kdeplot, zorder=0, n_levels=6))
        g.ax_joint.plot([min(prediction) - 0.1, max(prediction) + 0.1], [0, 0], linewidth=1, linestyle='--',
                        color=color)
        g.ax_marg_y.axhline(y=0, linewidth=1, linestyle='--', color=color)
        plt.xlabel('Predicted value', fontsize=15)
        plt.ylabel('Errors, tones/ha', fontsize=15)
        plt.show()

        g = (sns.jointplot('Prediction', 'Errors', kind="kde", data=plot_data, space=0, height=7,
                           color=color, alpha=0.2))
        g.set_axis_labels('Predicted value', 'Errors, tones/ha', fontsize=15)
        g.ax_joint.plot([min(prediction) - 0.1, max(prediction) + 0.1], [0, 0], linewidth=1, linestyle='--',
                        color=color)
        g.ax_marg_y.axhline(y=0, linewidth=1, linestyle='--', color=color)
        plt.show()



# Применение модели
# Wheat (tonnes per hectare)
# Rice (tonnes per hectare)
# Maize (tonnes per hectare)
# Barley (tonnes per hectare)
# France Germany Italy Romania Spain Czech Republic Netherlands Switzerland Austria Poland
# country = 'Poland'
# crop = 'Barley (tonnes per hectare)'
#
# reals = []
# preds = []
# ys = []
# # 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018
# for yEaR in [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
#     try:
#         pred, real = predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid', yEaR, country, crop, random_val = 20)
#     except ValueError:
#         try:
#             pred, real = predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid', yEaR, country, crop, random_val = 10)
#         except ValueError:
#             pred, real = predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid', yEaR, country, crop, random_val = 0)
#     reals.append(real)
#     preds.append(pred)
#     ys.append(yEaR)
#
# ys = np.array(ys)
# reals = np.array(reals)
# preds = np.array(preds)
# print_metrics(reals, preds)
# residuals_plots(reals, preds, color = 'green')
#
# # Переводим все в одномерный массивы
# ys = np.ravel(ys)
# reals = np.ravel(reals)
# preds = np.ravel(preds)
# # Теперь необходимо сохранить предсказания в файл
# df = pd.DataFrame({'Year': ys,
#                    'Prediction': preds,
#                    'Real': reals})
#
# file_name = 'Bayesian_' + country + '_' + crop + '.csv'
# file_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', file_name)
# # Сохранять пока не будем
# df.to_csv(file_path, sep=';', encoding='utf-8', index = False)