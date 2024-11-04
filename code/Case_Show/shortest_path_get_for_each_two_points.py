import pandas as pd
import argparse
import heapq
import numpy as np
import csv



parser = argparse.ArgumentParser()
parser.add_argument('--AB_img_path',type=str,default='/data/Brain Tumor2/testAB_img/') ###images involved in the nodes of the topology graph are all placed in this folder
parser.add_argument('--Nodes_codes_center_save_path',type=str,default='/code/Case_Show/results/Nodes_codes_center.csv')
parser.add_argument('--Nodes_connection_distance_save_path',type=str,default='/code/Case_Show/results/Nodes_connection_distance.csv')
parser.add_argument('--image_class_associated_codes_path',type=str,default='/code/CL_Analysis/results/testAB_CL_codes_extraction_results.csv')
parser.add_argument('--shortest_path_save_path',type=str,default='/results/AB_images_shortest_path_save.csv')
opts = parser.parse_args()

##For those pair points(images) whose shortest paths will be calculated, the start points(images) are listed in the "A_img_name_list" while the end points(images) are listed in the "B_img_name_list"
###For example
A_img_name_list=['z_86_01567.png']
B_img_name_list=['z_86_01318.png']

##For each node in the graph, the center vector(mean values) of all class-association codes involved into this node were recorded in this file
Nodes_codes_center_save_path=opts.Nodes_codes_center_save_path

##The distance between each two nodes in the graph were recorded in this file
Nodes_connection_distance_save_path=opts.Nodes_connection_distance_save_path

##The class-associated codes of the images were recorded in this file
image_class_associated_codes_path=opts.image_class_associated_codes_path

##For each two images(points), their corresponding nodes and the shortest paths will be recorded in this file
shortest_path_save_path=opts.shortest_path_save_path


AB_img_path=opts.AB_img_path

def read_graph_from_excel(file_name):
    df = pd.read_excel(file_name, index_col=0)
    graph = {}
    for vertex1 in df.index:
        for vertex2, weight in df[vertex1].items():
            if pd.isna(weight):
                continue
            if vertex1 not in graph:
                graph[vertex1] = {}
            if vertex2 not in graph:
                graph[vertex2] = {}
            graph[vertex1][vertex2] = weight
            graph[vertex2][vertex1] = weight
    return graph

def get_path(previous_vertices, start, end):
    path = []
    current_vertex = end
    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = previous_vertices[current_vertex]
    if path[-1] != start:
        return []
    path.reverse()
    return path

def dijkstra(graph, start, end):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    previous_vertices = {vertex: None for vertex in graph}
    priority_queue = [(0, start)]
    while priority_queue:
        (current_distance, current_vertex) = heapq.heappop(priority_queue)
        if current_vertex == end:
            return (distances[end], get_path(previous_vertices, start, end))
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))
    return (float('infinity'), [])



Nodes_connection_distance_save_file=pd.read_csv(Nodes_connection_distance_save_path,encoding='utf-8')
Nodes_connection_distance_save_file.to_excel(Nodes_connection_distance_save_path.replace('.csv','.xlsx'),index=False, engine='openpyxl')
Nodes_codes_center_file=pd.read_csv(Nodes_codes_center_save_path)

image_class_associated_codes_data=pd.read_csv(image_class_associated_codes_path)

graph = read_graph_from_excel(Nodes_connection_distance_save_path.replace('.csv','.xlsx'))


with open(shortest_path_save_path,"w",newline="") as csvfile:
    writer = csv.writer(csvfile)
    # columns_name
    columns_name_list = ['A(start_img)','B(end_img)','A(start_node)','B(end_node)','A2B_shortest_path','A2B_shortest_distance']
    writer.writerow(columns_name_list)

    for i in range(len(A_img_name_list)):
        A_img_name=A_img_name_list[i]
        B_img_name = B_img_name_list[i]

        data_A_CS=image_class_associated_codes_data.loc[image_class_associated_codes_data['image_name'] == A_img_name].values.tolist()[0][1:9]
        data_A_CS_numpy=np.array(data_A_CS)

        data_B_CS = image_class_associated_codes_data.loc[image_class_associated_codes_data['image_name'] == B_img_name].values.tolist()[0][1:9]
        data_B_CS_numpy = np.array(data_B_CS)


        for j in range(len(Nodes_codes_center_file)):

            Nodes_codes_center_numpy=np.array(Nodes_codes_center_file.loc[j][1:].values.tolist())
            A_distance_j=np.sqrt(np.sum(np.square(Nodes_codes_center_numpy - data_A_CS_numpy)))
            B_distance_j = np.sqrt(np.sum(np.square(Nodes_codes_center_numpy - data_B_CS_numpy)))

            if j == 0:
                A_node = Nodes_codes_center_file.loc[0][0]
                A_distance_new = A_distance_j
                A_node_codes_center = Nodes_codes_center_numpy
                B_node = Nodes_codes_center_file.loc[0][0]
                B_distance_new = B_distance_j
                B_node_codes_center = Nodes_codes_center_numpy


            if j>0:
                if A_distance_j<=A_distance_new:
                    A_node =Nodes_codes_center_file.loc[j][0]
                    A_distance_new =A_distance_j
                    A_node_codes_center=Nodes_codes_center_numpy

                if B_distance_j<=B_distance_new:
                    B_node =Nodes_codes_center_file.loc[j][0]
                    B_distance_new =B_distance_j
                    B_node_codes_center = Nodes_codes_center_numpy


        distance_AB, path_AB = dijkstra(graph, A_node, B_node)

        path_str=''
        for k in range(len(path_AB)):
            if k!=len(path_AB)-1:
                path_str=path_str+path_AB[k]+','
            if k == len(path_AB) - 1:
                path_str = path_str + path_AB[k]

        row_list=[A_img_name,B_img_name,A_node,B_node,path_str,str(distance_AB)]
        writer.writerow(row_list)

