df_test = df_test[["x", "y"]]

df_coordinates = dataset.df[["x", "y"]]
current_point = df_test[['x', 'y']]
distance_array = distance_matrix(df_coordinates, current_point)
distance_df = pd.DataFrame(distance_array)
distance_df = distance_df.T
distance_df


# for each test point create a neighbourhood
prediction_neighbourhoods = []
for station in range(df_test.shape[0]):
    current_station = station
    # print(current_station)
    list_neighbourhood_temp = []
    for i in range(neigh_size):
        index1 = int(current_station)
        index2 = int(distance_df.iloc[current_station].sort_values().index[i])
        distance = distance_df.iloc[current_station].sort_values()[index2]
        ln2 = dataset.df["log_normal"][index2]
        list_for_df = [index1, index2, distance, ln2]
        list_neighbourhood_temp.append(list_for_df)
        # print(list_neighbourhood_temp)
    new_neigh_df = pd.DataFrame(list_neighbourhood_temp, columns=[
        "index1", "index2", "distance", "ln2"])
    # print(new_neigh_df)
    prediction_neighbourhoods.append(new_neigh_df)

# Step 1: find all the distances between current test point and all train points
# step 2: sort the distances
# step 3: find the closest neighbour
# step 4: build the neighbourhood for the closest neighbour


# #for curr_neigh in dataset_test.neighbourhoods:
# list_neighbourhood_temp = []
# curr_neigh = dataset_test.neighbourhoods[0]

# curr_neigh.sort_values(by=['distance'], inplace=True)
# closest_neighbour = curr_neigh.iloc[0]["index2"]
# curr_neigh_temp = curr_neigh[['index2', 'rank2']]
# for i in range(1, curr_neigh_temp.shape[0]):
#     index1 = int(curr_neigh_temp.iloc[[0]]["index2"])
#     index2 = int(curr_neigh_temp.iloc[[i]]["index2"])

#     list_for_df = [index1, index2,
#                     dataset_test.distance_df[index1][index2]]
#     list_neighbourhood_temp.append(list_for_df)
# new_neigh_df = pd.DataFrame(list_neighbourhood_temp, columns=[
#     "index1", "index2", "distance"])
# display(new_neigh_df)
# #list_neighbourhood_updated.append(new_neigh_df)

# for curr_neigh in dataset_test.neighbourhoods:
#     list_neighbourhood_temp = []
#     curr_neigh.sort_values(by=['distance'], inplace=True)
#     closest_neighbour = curr_neigh.iloc[0]["index2"]
#     curr_neigh_temp = curr_neigh[['index2', 'rank2']]
#     for i in range(1, curr_neigh_temp.shape[0]):
#         index1 = int(curr_neigh_temp.iloc[[0]]["index2"])
#         index2 = int(curr_neigh_temp.iloc[[i]]["index2"])

#         list_for_df = [index1, index2,
#                         dataset_test.distance_df[index1][index2]]
#         list_neighbourhood_temp.append(list_for_df)
#     new_neigh_df = pd.DataFrame(list_neighbourhood_temp, columns=[
#         "index1", "index2", "distance"])
#     list_neighbourhood_updated.append(new_neigh_df)