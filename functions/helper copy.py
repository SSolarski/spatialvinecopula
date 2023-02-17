import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skgstat as skg
from scipy.spatial import distance_matrix
from scipy import stats
from scipy.stats import lognorm
from scipy import integrate
import pyvinecopulib as pv

# Calculate rank of variable of interest


class DataSet:
    def __init__(self, df, variable_of_interest):
        self.df = df
        self.number_of_stations = len(df)
        self.variable = variable_of_interest
        self.distance_array = None
        self.distance_df = None
        self.cutoff_value = None
        self.pairs_cutoff = None
        self.all_pairs = None

    def add_rank(self):
        rank = self.df.rank()[self.variable]
        rank = rank/(len(rank)+1)
        self.df["rank"] = rank

    def construct_pairs(self):

        # calculate distances between all pairs of points
        df_coordinates = self.df[["x", "y"]]
        self.distance_array = distance_matrix(df_coordinates, df_coordinates)
        list_of_indexes = np.argwhere(self.distance_array > 0)

        # remove duplicate rows
        list_nodupe = []
        for row in list_of_indexes:
            # ensure there are no duplicates
            if (row[0] < row[1]):
                list_nodupe.append(row)
        len(list_nodupe)

        # now we are working with the whole dataset (not cutoff)
        df_whole = pd.DataFrame(list_nodupe, columns=["index1", "index2"])

        distance_value = []

        for _, row in df_whole.iterrows():
            distance_value.append(
                self.distance_array[row["index1"]][row["index2"]])

        df_whole["rank1"] = list(self.df["rank"][(df_whole["index1"])])
        df_whole["rank2"] = list(self.df["rank"][(df_whole["index2"])])
        df_whole["distance"] = distance_value

        self.all_pairs = df_whole

    # Convert the original dataframe to pairs of stations

    def apply_cutoff(self, cutoff_value=1200):

        self.cutoff_value = cutoff_value
        self.pairs_cutoff = self.all_pairs.drop(
            self.all_pairs[self.all_pairs['distance'] > cutoff_value].index)


class SpatialCopula:

    def __init__(self, dataset, num_bins=10, cutoff_value=1200):
        """Initialize the class for the first spatial tree"""
        self.dataset = dataset
        self.num_bins = num_bins
        self.cutoff_value = cutoff_value
        self.bin_means_list = None
        self.bins_data = None
        self.kendall_list = None
        self.degree = 1
        self.predict = None
        self.bin_means_cut = None
        self.kendall_list_predict = None
        self.copulas = None
        self.neigh_list = None

    def calculate_bins(self):
        """Calculate the bins for the distance values in the first tree"""
        bin_values, _ = pd.cut(self.dataset.pairs_cutoff["distance"], bins=self.num_bins,
                               labels=range(0, self.num_bins),
                               retbins=True)
        self.dataset.pairs_cutoff["bins"] = bin_values

        # find the mean value for each bin
        bin_means = self.dataset.pairs_cutoff[[
            "distance", "bins"]].groupby('bins').mean()

        # Store the means in a list
        self.bin_means_list = np.concatenate((bin_means.values)).tolist()

        # make separate dataframes for each bin
        bins_data = []
        for i in range(0, self.num_bins):
            bins_data.append(
                self.dataset.pairs_cutoff[self.dataset.pairs_cutoff["bins"] == (i)])
        # print(bins_data[0].rename(columns={"index1": "center_id", "index2": "neigh_id",
        #                                    "rank1": "center_rank", "rank2": "neigh_rank"}).head())
        self.bins_data = bins_data
        print(f"Bin means: {self.bin_means_list}")

    def calculate_kendall(self):

        kendall_list = []
        for bin_data in self.bins_data:
            kendall_list.append(stats.kendalltau(
                bin_data["rank1"], bin_data["rank2"])[0])

        self.kendall_list = kendall_list

    def fit_model_kendall(self, degree=1):

        self.degree = degree

        # replace nan values with 0
        list(pd.Series(self.kendall_list).replace(np.nan, 0))

        # create a dataframe with the bin means and kendall's tau
        df_temp = pd.DataFrame(
            {'bin_mean': self.bin_means_list, 'kendall': self.kendall_list})

        # Check if there are any negative values in the kendall's tau
        if (len(self.bin_means_list) > 1):
            if (df_temp[df_temp['kendall'] <= 0].empty == False and self.degree == 1):
                first_index = df_temp[df_temp['kendall'] <= 0].index[0]
                df_temp = df_temp.drop(df_temp.index[first_index:])

        # fit a polynomial to the kendall's tau
        model = np.polyfit(df_temp['bin_mean'], df_temp['kendall'], degree)
        self.predict = np.poly1d(model)

        # predict the kendall's tau for each bin
        self.kendall_list_predict = self.predict(df_temp['bin_mean'])

        # save the bin means for the fitted polynomial (without the negative values of kendall's tau)
        self.bin_means_cut = df_temp['bin_mean']

    def plot_kendall(self):
        # find the root of the polynomial
        if self.degree == 1:
            root = int(self.predict.roots)
        else:
            root = self.dataset.cutoff_value

        # plot the polynomial
        x_lin_reg = range(0, root)
        y_lin_reg = self.predict(x_lin_reg)

        # find the point at which the pedict(x_lin_reg)==0
        x_lin_reg2 = range(root, self.dataset.cutoff_value)
        y_lin_reg2 = np.zeros(len(x_lin_reg2))

        # plot the kendall's tau if there are more than 2 points
        if (len(self.bin_means_list) > 2):
            plt.xlabel("Distance")
            plt.ylabel("Kendall's Tau")

            plt.scatter(self.bin_means_list, self.kendall_list)
            plt.plot(x_lin_reg, y_lin_reg, c='green')
            plt.plot(x_lin_reg2, y_lin_reg2, c='green')

    def calculate_copulas(self, family_set=None):
        """Convert the bins to copulas and set the parameters to the optimal predicted parameters"""

        if family_set is None:
            family_set = [pv.BicopFamily.indep, pv.BicopFamily.gaussian,
                          pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe]

        # empty list to store the copulas
        cops = []

        # loop over the bins
        for i in range(0, len(self.bin_means_cut)):

            # define the data u1 and u2 from the ranks
            rank1 = np.array(self.bins_data[i]["rank1"])
            rank2 = np.array(self.bins_data[i]["rank2"])
            data = np.stack((rank1, rank2), axis=0).T

            # define the family set and fitting method
            controls = pv.FitControlsBicop(family_set=family_set,
                                           parametric_method="itau")

            # append the current copula to the list of copulas
            cops.append(pv.Bicop(data=data, controls=controls))

            # set the parametrs to the optimal parameter according
            # to the predicted kendall's tau
            cops[i].parameters = cops[i].tau_to_parameters(
                self.kendall_list_predict[i])

        # append an independent copula if there is only one bin
        if (len(self.kendall_list_predict) > 1):
            cops.append(pv.Bicop(pv.BicopFamily.indep))

        self.copulas = cops

    # build the neighbourhood for each station

    def build_neighbourhoods(self, neighbourhood_size=20):

        neigh_list = []
        for i in range(0, self.dataset.number_of_stations):
            curr_df = self.dataset.all_pairs[self.dataset.all_pairs["index1"]
                                             == (i)].sort_values(by=['distance'])
            curr_df = curr_df.reset_index(drop=True)
            neigh_list.append(curr_df.head(neighbourhood_size))

        self.neighbourhoods = neigh_list


# calculate which bin a distance belongs to
def distance_to_bin(distance, bin_means_list):
    if (len(bin_means_list) > 1):
        iwidth = bin_means_list[1] - bin_means_list[0]
    else:
        return (0)
    bin_result = np.searchsorted(bin_means_list, distance, side='right')

    return(bin_result)


# calculate the copula indexes according to which bin it belongs to
def bin_to_copulas(bin_result, bin_means_list, curr_distance):

    # find copula indx and lambda from the bin result
    if (bin_result == 0):
        copula_idx = (0, 0)
        curr_lambda = 1
    elif (bin_result == len(bin_means_list)):
        if (len(bin_means_list) > 1):
            copula_idx = (len(bin_means_list), len(bin_means_list))
        else:
            copula_idx = (0, 0)

        curr_lambda = 1
        #print(curr_distance,bin_result, bin_means_list[len(bin_means_list)-1])
        # print(copula_idx)
    else:
        copula_idx = (bin_result - 1, bin_result)
        # print(bin_result)
        # print(bin_means_list)
        curr_lambda = (curr_distance - bin_means_list[copula_idx[0]]) / (
            bin_means_list[copula_idx[1]] - bin_means_list[copula_idx[0]])

    # copula_idx is a tuple (0,1) and curr_lamba is a number
    return copula_idx, curr_lambda

# calculate the hfunc value for given copulas and lambda


def calc_hfunc(curr_lambda, copula1, copula2, row):
    hfunc_value = ((1 - curr_lambda) * copula1.hfunc1([[row["rank1"], row["rank2"]]])[0])  \
        + (curr_lambda * copula2.hfunc1([[row["rank1"], row["rank2"]]])[0])

    return hfunc_value

# build the next neighbourhood from the current neighbourhood


def build_next_neighbourhood(list_neighbourhood, bin_means_list, cops, model, bin_means_cut, distance_df):

    list_neighbourhood_updated = []

    for neigh in list_neighbourhood:

        # pass in the rows, one by one in the hfucn1
        list_hfunc = []
        for i, row in neigh.iterrows():

            # check which bin the row belongs in
            bin_result = distance_to_bin(row["distance"], bin_means_cut)

            # calculate copulaidx and lambda
            copulas_idx, curr_lambda = bin_to_copulas(
                bin_result, bin_means_cut, row["distance"])

            copula1, copula2 = cops[copulas_idx[0]], cops[copulas_idx[1]]

            # update the copula parameters with the model for tau
            copula1.parameters = copula1.tau_to_parameters(
                np.maximum(0, model(row["distance"])))
            copula2.parameters = copula2.tau_to_parameters(
                np.maximum(0, model(row["distance"])))

            if copulas_idx[0] == (len(bin_means_cut)-1):
                # print(copula1)
                pass

            # now use the copulas to calculate hfunc
            hfunc_value = calc_hfunc(curr_lambda, copula1, copula2, row)
            if curr_lambda > 1:
                print(curr_lambda, copulas_idx, row)
                print(bin_means_list[copulas_idx[0]],
                      bin_means_list[copulas_idx[1]])
                print("error")

            # append to current neighbourhood list
            list_hfunc.append(hfunc_value)

        # add hfunc column
        neigh["hfunc"] = list_hfunc

        # drop everything but the index of the neighbour and value of hfunc
        neigh_temp = neigh[["index2", "hfunc"]]

# build new neighbourhood
        list_neighbourhood_temp = []

        for i in range(1, neigh_temp.shape[0]):
            index1 = int(neigh_temp.iloc[[0]]["index2"])
            index2 = int(neigh_temp.iloc[[i]]["index2"])
            rank1 = float(neigh_temp.iloc[[0]]["hfunc"])
            rank2 = float(neigh_temp.iloc[[i]]["hfunc"])

            list_for_df = [index1, index2,
                           distance_df[index1][index2], rank1, rank2]
            list_neighbourhood_temp.append(list_for_df)

        new_neigh_df = pd.DataFrame(list_neighbourhood_temp, columns=[
                                    "index1", "index2", "distance", "rank1", "rank2"])
        list_neighbourhood_updated.append(new_neigh_df)

    return(list_neighbourhood_updated)


# calculate the bins for the new neighbourhood
def list_neigh_to_bin_mean(list_neighbourhood_temp, cutoff_value_temp, num_bins=10, degree=1):
    df_whole_temp = pd.DataFrame()
    for neigh in list_neighbourhood_temp:
        df_whole_temp = pd.concat([df_whole_temp, neigh], ignore_index=True)

    #cutoff_value_temp = 600
    df_whole_temp_cut = df_whole_temp[df_whole_temp["distance"]
                                      < cutoff_value_temp]

    #num_bins = 10
    bin_values, _ = pd.cut(df_whole_temp_cut["distance"], bins=num_bins,
                           labels=range(0, num_bins),
                           retbins=True)

    df_whole_temp_cut.loc[:, ("bins")] = list(bin_values)
    bin_means = df_whole_temp_cut[["distance", "bins"]].groupby('bins').mean()

    bins_data = []
    kendall_list = []

    for i in range(0, num_bins):
        bins_data.append(df_whole_temp_cut[df_whole_temp_cut["bins"] == (i)])
        kendall_list.append(stats.kendalltau(df_whole_temp_cut[df_whole_temp_cut["bins"] == (i)]["rank1"],
                                             df_whole_temp_cut[df_whole_temp_cut["bins"] == (i)]["rank2"])[0])

    bin_means_list = np.concatenate((bin_means.values)).tolist()

    model_predict, kendall_list_predict, bin_means_cut = fit_model_kendall(
        bin_means_list, kendall_list, cutoff_value_temp, degree)

    return(bins_data, bin_means_list, kendall_list_predict, model_predict, bin_means_cut, kendall_list)


# calculate the copulas for the new neighbourhood
def spcopula_step(list_neighbourhood_curr, bin_means_curr, cops_curr, model_curr, bin_means_cut_curr,
                  distance_df, family_set, cutoff=600, num_bins=10, degree=3):

    list_neighbourhood_next = build_next_neighbourhood(
        list_neighbourhood_curr, bin_means_curr, cops_curr, model_curr, bin_means_cut_curr, distance_df)
    bins_data_next, bin_means_list_next, kendall_list_predict_next, model_predict_next, bin_means_cut_next, kendall_list = list_neigh_to_bin_mean(
        list_neighbourhood_next, cutoff, num_bins, degree)
    cops_next = bins_to_cop(bins_data_next, family_set,
                            kendall_list_predict_next, bin_means_cut_next)

    return(list_neighbourhood_next, bin_means_list_next, cops_next, model_predict_next, bin_means_cut_next, kendall_list)


def build_vine_copula(super_list_neighbourhood, super_bin_means_list, super_cops, super_model_list, super_bin_means_cut, super_kendall_list, distance_df, cutoff_list, num_bins_list, degree_list, family_set, neigh_size):

    for i in range(1, neigh_size):
        super_list_neighbourhood_temp, super_bin_means_list_temp, super_cops_temp, super_model_temp, super_bin_means_cut_temp, super_kendall_list_temp = \
            spcopula_step(super_list_neighbourhood[i-1], super_bin_means_list[i-1],
                          super_cops[i-1], super_model_list[i -
                                                            1], super_bin_means_cut[i - 1],
                          cutoff=cutoff_list[i], num_bins=num_bins_list[i],
                          degree=degree_list[i], family_set=family_set, distance_df=distance_df)

        super_list_neighbourhood.append(super_list_neighbourhood_temp)
        super_bin_means_list.append(super_bin_means_list_temp)
        super_cops.append(super_cops_temp)
        super_model_list.append(super_model_temp)
        super_bin_means_cut.append(super_bin_means_cut_temp)
        super_kendall_list.append(super_kendall_list_temp)
        print(i)
    return(super_list_neighbourhood, super_bin_means_list, super_cops, super_model_list, super_bin_means_cut, super_kendall_list)


def add_lognormal(df):
    #  build a log normal, so that we have a different representation of zinc for testing
    #ppf(q, s, loc=0, scale=1)

    ln_mean = np.mean(np.log(df["zinc"]))
    ln_std = np.std(np.log(df["zinc"]))
    ln_mean_exp = np.exp(ln_mean)

    frozen_lognorm = lognorm(s=ln_std, scale=ln_mean_exp)
    log_normal = frozen_lognorm.cdf(df["zinc"])
    df["log_normal"] = log_normal
    return(df, frozen_lognorm)


def construct_pairs_lognorm(df):

    df_coordinates = df[["x", "y"]]
    distance_array = distance_matrix(df_coordinates, df_coordinates)
    list_of_indexes = np.argwhere(distance_array > 0)

    # now we are working with the whole dataset (not cutoff)
    df_whole_ln = pd.DataFrame(list_of_indexes, columns=["index1", "index2"])

    distance_value = []

    for index, row in df_whole_ln.iterrows():
        distance_value.append(distance_array[row["index1"]][row["index2"]])

    df_whole_ln["ln1"] = list(df["log_normal"][(df_whole_ln["index1"])])
    df_whole_ln["ln2"] = list(df["log_normal"][(df_whole_ln["index2"])])
    df_whole_ln["distance"] = distance_value
    df_whole_ln.head()

    return(df_whole_ln)


def distances_per_tree(super_list_neighbourhood, neigh_size, number_of_stations):
    dist_df_list = []
    i = neigh_size
    spDepth = neigh_size
    for list_neigh in super_list_neighbourhood[0:spDepth]:
        temp_list = []
        for neigh in list_neigh:
            temp_list.append(list(neigh[0:i]["distance"]))
        dist_df_list.append(pd.DataFrame(temp_list))
        i -= 1

    # calculate distances
    h_temp_small = []
    h_big = []
    for i in range(number_of_stations):
        for dataFrame in dist_df_list:
            h_temp_small.append(dataFrame.iloc[i])
        h_big.append(h_temp_small)
        h_temp_small = []

    return (h_big)


def build_xvalue(n):
    rat_temp = np.array([[1e-06, 1e-05, 1e-04, 1e-03]])
    rat_temp2 = np.array([[x for x in range(1, 51)]])
    rat = rat_temp.T*rat_temp2
    rat_temp3 = np.array([x/n for x in range(1, n)])
    rat_inv = 1-rat
    rat_final = np.append(rat, rat_inv)
    rat_final = np.append(rat_final, rat_temp3)
    xvalue = np.sort(np.unique(rat_final.flatten()))

    return xvalue


def calc_hfunc_list(curr_lambda, copula1, copula2, u0temp):
    hfunc_value = ((1 - curr_lambda) * copula1.hfunc1(u0temp))  \
        + (curr_lambda * copula2.hfunc1(u0temp))
    return hfunc_value


def calc_pdf(curr_lambda, copula1, copula2, u0temp):
    pdf_value = ((1 - curr_lambda) * copula1.pdf(u0temp))  \
        + (curr_lambda * copula2.pdf(u0temp))

    return pdf_value


def dCopula(repCondVar, spVine, spDepth, models, super_bin_means_cut, h, nx):
    l0 = np.zeros(nx)
    u0 = repCondVar

    for spTree in range(0, spDepth):
        u1 = []
        curr_cops = spVine[spTree]
        curr_model = models[spTree]
        tmph = h[spTree]
        for i in range(0, len(tmph)):

            # calculate value to update l0
            u0temp = u0[:, [0, i+1]]

            curr_distance = tmph[i]
            # now we calculate the density of all pairs from u0temp
            # we use the bins and appropriate copulas + lambdas

            # check which bin the row belongs in
            bin_result = distance_to_bin(
                curr_distance, super_bin_means_cut[spTree])

            # calculate copulaidx and lambda
            copulas_idx, curr_lambda = bin_to_copulas(
                bin_result,  super_bin_means_cut[spTree], curr_distance)

            copula1, copula2 = curr_cops[copulas_idx[0]
                                         ], curr_cops[copulas_idx[1]]
            # update parameters
            copula1.parameters = copula1.tau_to_parameters(
                np.maximum(0, curr_model(curr_distance)))
            copula2.parameters = copula2.tau_to_parameters(
                np.maximum(0, curr_model(curr_distance)))

            # now use the copulas to calculate hfunc
            pdf_value = calc_pdf(curr_lambda, copula1, copula2, u0temp)
            hfunc_value = calc_hfunc_list(
                curr_lambda, copula1, copula2, u0temp)

            l0 = l0 + np.log(pdf_value)
            u1.append(hfunc_value)

        # now we want to calculate u1 from u0
        u1 = np.array(u1)
        u0 = u1.T

    return(np.exp(l0))


# to start with condSpVine we need the information contained in list_neighbourhood_ln and dist_df_list
def condSpVine(condVar,  h, spVine, models, spDepth, super_bin_means_cut, n=1000):
    # condVar is list neighbourhood

    xvalue = build_xvalue(n)
    nx = len(xvalue)

    repCondVar = np.append(np.reshape(xvalue, (len(xvalue), 1)), np.repeat(
        np.matrix(condVar), nx, axis=0), axis=1)

    density = dCopula(repCondVar, spVine, spDepth,
                      models, super_bin_means_cut, h, nx)
    left = max(0, 2*density[0] - density[1])
    right = max(0, 2*density[nx-1] - density[nx-2])
    density_extended = np.append(np.append(left, density), right)

    return density_extended


def calculate_predictions(list_neighbourhood_ln, df, super_cops, super_model_list,
                          super_bin_means_cut, neigh_size, frozen_lognorm, h_big,
                          number_of_stations
                          ):

    xvals = build_xvalue(1000)
    xvals_extended = np.append(np.append([0], xvals), [1])

    density_list = []
    integration_constant_list = []
    result_list = []
    count = 0
    error_list = []

    for i in range(number_of_stations):
        density = condSpVine(
            condVar=list_neighbourhood_ln[i]['ln2'], h=h_big[i],
            spVine=super_cops, models=super_model_list, spDepth=neigh_size,
            super_bin_means_cut=super_bin_means_cut
        )

        density_list.append(density)
        integration_constant = integrate.simpson(density, xvals_extended)
        integration_constant_list.append(integration_constant)

        result = integrate.simpson(((frozen_lognorm.ppf(
            xvals) * density[1:len(xvals)+1]) / integration_constant), xvals)
        result_list.append(result)
        print("Data point " + str(i))
        if (result > 2000):
            print(i, result)
            error_list.append(result)
            count += 1

    df_result = pd.DataFrame()
    df_result["zinc"] = df["zinc"]
    df_result["result"] = (result_list)
    #df_result["result"] = df_result["result"]
    df_result = df_result[~df_result['result'].isin(error_list)]

    print("Final Result: " + str(np.median(abs(result_list - df["zinc"]))))
    print("Number of errors: " + str(count))

    return(df_result)


def plot_original_data(df_result):
    plt.scatter(df_result["x"], df_result["y"],
                c=df_result["zinc"], cmap=plt.cm.gnuplot2)
    # get current axes
    ax = plt.gca()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original Data")
    # hide x-axis
    ax.get_xaxis().set_visible(False)

    # hide y-axis
    ax.get_yaxis().set_visible(False)

    plt.colorbar()
    plt.show()


def plot_predicted_data(df_result):
    plt.scatter(df_result["x"], df_result["y"],
                c=df_result["result"], cmap=plt.cm.gnuplot2)
    # get current axes
    ax = plt.gca()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Prediction")

    # hide x-axis
    ax.get_xaxis().set_visible(False)

    # hide y-axis
    ax.get_yaxis().set_visible(False)

    plt.colorbar()
    plt.show()


def plot_result_statistics(df_result):
    plt.scatter(df_result["zinc"], df_result["result"])
    plt.plot([0, 2000], [0, 2000], 'r')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

    plt.hist(df_result["zinc"] - df_result["result"], bins=50)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title("Error Histogram")
    plt.show()

    plt.scatter(df_result["zinc"], df_result["zinc"] - df_result["result"])
    plt.xlabel("Actual")
    plt.ylabel("Error")
    plt.title("Actual vs Error")
    plt.show()

    plt.scatter(df_result["result"], df_result["zinc"] - df_result["result"])
    plt.xlabel("Predicted")
    plt.ylabel("Error")
    plt.title("Predicted vs Error")
    plt.show()

    print("Median Absolute Error: " +
          str(np.median(abs(df_result["zinc"] - df_result["result"]))))
    print("Mean Absolute Error: " +
          str(np.mean(abs(df_result["zinc"] - df_result["result"]))))
    print("Mean Squared Error: " +
          str(np.mean((df_result["zinc"] - df_result["result"])**2)))
    print("Root Mean Squared Error: " +
          str(np.sqrt(np.mean((df_result["zinc"] - df_result["result"])**2))))
