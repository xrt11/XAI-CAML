1. For one new dataset, we are supposed to run the "main_train.py" file to get the new generative models.
2. After training the model, we can employ the trained model and the "topology_graph_get.py" file for performing the topological data analysis on the class-associated codes extracted from the new dataset.
3. When we get the topology graph of the dataset, the distances between each two nodes within the topology graph will be calculated by running the procedure designed in the "graph_nodes_distance_matrix.py".
4. By employing the distance matrix(where the distances between each two nodes in the topology graph were recorded), we can use the procedure designed in the "shortest_path_get_for_each_two_points.py" file to generate the shortest path from the starting point/image to the ending point/image.
5. And we can perform counterfactual generation along the shortest paths designed above by executing the procedure designed in the "counterfactual_generation_along_path.py" file.
