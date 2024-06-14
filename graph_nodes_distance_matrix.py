import pandas as pd
import heapq
import os
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import argparse
import numpy.linalg as la
from trainer_exchange import trainer
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--Nodes_images_path',type=str,default='Nodes_images_name.txt')
parser.add_argument('--Nodes_connection_save_path',type=str,default='Nodes_connection.txt')
parser.add_argument('--AB_img_path',type=str,default='testAB_img/') ###images involved in the nodes of the topology graph are all placed in this folder
parser.add_argument('--Nodes_codes_center_save_path',type=str,default='Nodes_codes_center.csv')
parser.add_argument('--Nodes_all_distance_save_path',type=str,default='Nodes_all_distance.csv')
parser.add_argument('--Nodes_connection_distance_save_path',type=str,default='Nodes_connection_distance.csv')
parser.add_argument('--CAML_trained_gen_model_path',type=str,default='models/gen_00150000.pt')
parser.add_argument('--style_dim',type=int,default=8)
parser.add_argument('--train_is',type=str,default='False')
opts = parser.parse_args()

if opts.cuda=='True':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


Nodes_images_path=opts.Nodes_images_path
Nodes_connection_save_path=opts.Nodes_connection_save_path
Nodes_codes_center_save_path=opts.Nodes_codes_center_save_path
Nodes_all_distance_save_path=opts.Nodes_all_distance_save_path
Nodes_connection_distance_save_path=opts.Nodes_connection_distance_save_path


def nodes_connection_get(Nodes_images_path=None,Nodes_connection_save_path=None):
    with open(Nodes_images_path)as file1:
        linse1=file1.readlines()

    ID_IMG_list={}
    ID_all_list=[]
    for i in range(len(linse1)):
        img_list=[]
        for j in range(1,len(linse1[i].strip().split())):

            if linse1[i].strip().split()[j] not in img_list:
                img_list.append(linse1[i].strip().split()[j])

        ID_IMG_list.update({linse1[i].strip().split()[0]:img_list})
        ID_all_list.append(linse1[i].strip().split()[0])

    file2=open(Nodes_connection_save_path,'a')
    for i in range(len(linse1)):
        id_have=[]
        for j in range(1,len(linse1[i].strip().split())):

            for k in range(i+1,len(ID_all_list)):

                if linse1[i].strip().split()[j] in ID_IMG_list[ID_all_list[k]]:
                    if ID_all_list[k] not in id_have:
                        id_have.append(ID_all_list[k])
                        file2.write(linse1[i].strip().split()[0]+','+ID_all_list[k])
                        file2.write('\n')
    file2.close()

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


def compute_squared_EDM_method(X):
  n,m = X.shape
  D = np.zeros([n, n])
  for i in range(n):
    for j in range(i+1, n):
      D[i,j] = la.norm(X[i, :] - X[j, :])
      D[j,i] = D[i,j]
  return D

###generator model loading
gen_model_path = opts.CAML_trained_gen_model_path
train_is=opts.train_is
style_dim=opts.style_dim
trainer = trainer(device=device,style_dim=style_dim,optim_para=None,gen_loss_weight_para=None,dis_loss_weight_para=None,train_is=train_is)
trainer.to(device)
state_dict_gen = torch.load(gen_model_path, map_location=device)
trainer.gen.load_state_dict(state_dict_gen['ab'])
trainer.eval()
encode = trainer.gen.encode

####data preprocessing
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.CenterCrop((256, 256))] + transform_list
transform_list = [transforms.Resize(256)] + transform_list
transform = transforms.Compose(transform_list)

####class-associated codes extraction
img_n=0
s_tensor_all_dict={}
AB_img_path=opts.AB_img_path
with torch.no_grad():
    for img_name in os.listdir(AB_img_path):
        example_path = AB_img_path + img_name
        example_img = Variable(transform(Image.open(example_path).convert('RGB')).unsqueeze(0).to(device))
        c_A_ori, s_A_ori = encode(example_img)
        s_tensor_all_dict.update({img_name: s_A_ori})
        img_n=img_n+1
        print('class-associated codes extraction: '+str(img_n))

####images exist on each node of the topology graph are recorded.
with open(Nodes_images_path)as file_id_img:
    lines_id_img=file_id_img.readlines()
id_image_dict={}
id_all_list=[]
for i in range(len(lines_id_img)):
    img_list=[]
    for j in range(len(lines_id_img[i].strip().split())):
        if j==0:
            key_vl=lines_id_img[i].strip().split()[j]
        if j>0:
            img_list.append(lines_id_img[i].strip().split()[j])
    id_all_list.append(key_vl)
    id_image_dict.update({key_vl:img_list})

# Nodes codes centers are recorded
with open(Nodes_codes_center_save_path,"w",newline="") as csvfile:
    writer = csv.writer(csvfile)
    # columns_name
    columns_name_list = ['Node_ID']
    for latent_ids in range(style_dim):
        columns_name_list.append('index_'+str(latent_ids))
    writer.writerow(columns_name_list)
    id_center_vector_dict = {}
    for j in range(len(id_all_list)):
        img_all_list = id_image_dict[id_all_list[j]]
        s_A_mean = 0
        s_A_mean_str = ''
        for img_name2 in img_all_list:
            s_A = s_tensor_all_dict[img_name2]
            s_A_mean = s_A + s_A_mean
        s_A_mean = s_A_mean / len(img_all_list)
        id_center_vector_dict.update({id_all_list[j]: s_A_mean})
        row_list = [id_all_list[j]]
        for k in range(s_A_mean.size(1)):
            row_list.append(str(s_A_mean[0][k, 0, 0].item()))
        writer.writerow(row_list)


print('Nodes codes centers save finished')

##Nodes codes centers distances are recorded
df = pd.read_csv(Nodes_codes_center_save_path)
df = df.drop(columns='Node_ID')
X = np.array(df)
D = compute_squared_EDM_method(X)
D = pd.DataFrame(D)
D.index=id_all_list
D.columns=id_all_list
D.to_csv(Nodes_all_distance_save_path)

print('Nodes codes centers distances finished')

####### for each two nodes without connection, delete the distance information (replaced with "None", which means that the distance between these two nodes is infinity)
# 1.Read the nodes all distance file
with open(Nodes_all_distance_save_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # next line
    distances = {row[0]: {header[i+1]: float(x) for i, x in enumerate(row[1:])} for row in reader}

# 2.Get a file (txt format) where each two nodes that share samples (connection created) in the topology graph are recorded.
# For example, for the first line in the file, "ID2,ID5" means that node ID2 and node ID5 share samples and they are connected.
nodes_connection_get(Nodes_images_path=Nodes_images_path,Nodes_connection_save_path=Nodes_connection_save_path)

# 3.Read the connection between nodes
with open(Nodes_connection_save_path, 'r') as txtfile:
    reachable = set()
    for line in txtfile:
        p1, p2 = line.strip().split(',')
        reachable.add((p1, p2))
        reachable.add((p2, p1))  # p1 is related to p2 means that p2 is also related to p1

# 4.Delete the distance information for each two nodes without connection
for p1 in distances:
    for p2 in distances[p1]:
        if (p1 == p2):
            distances[p1][p2] = distances[p2][p1] = 0
        elif ((p1, p2) in reachable) or ((p2, p1) in reachable):
            distances[p1][p2] = distances[p2][p1] = distances[p1][p2]
        else:
            distances[p1][p2] = distances[p2][p1] = None

# 5. Save the final nodes connection distance file
with open(Nodes_connection_distance_save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([''] + list(distances.keys()))
    for p1 in distances:
        row = [p1] + [distances[p1][p2] for p2 in distances]
        writer.writerow(row)


print('Nodes codes centers connection distances finished')

