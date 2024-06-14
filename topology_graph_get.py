import os
import torch
from torch.autograd import Variable
from PIL import Image
from trainer_exchange import trainer
from torchvision import transforms
import argparse
import csv
import numpy as np
import kmapper as km
import sklearn
import matplotlib.pyplot as plt
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--A_img_path',type=str,default='testA_img/') ###testing images with class A are placed into this folder
parser.add_argument('--B_img_path',type=str,default='testB_img/') ###testing images with class B are placed into this folder
parser.add_argument('--AB_image_name_label_path',type=str,default='testAB_img-name_label.txt')
parser.add_argument('--latentAB_save_path',type=str,default='testAB_latent.csv')
parser.add_argument('--html_name01',type=str,default='topology_graph_custom_image_name.html')
parser.add_argument('--html_name02',type=str,default='topology_graph_custom_labels.html')
parser.add_argument('--CAML_trained_gen_model_path',type=str,default='models/gen_00150000.pt')
parser.add_argument('--style_dim',type=int,default=8)
parser.add_argument('--train_is',type=str,default='False')


opts = parser.parse_args()

if opts.cuda=='True':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

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
decode = trainer.gen.decode

####data preprocessing
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.CenterCrop((256, 256))] + transform_list
transform_list = [transforms.Resize(256)] + transform_list
transform = transforms.Compose(transform_list)

####image label get
AB_image_name_label_path=opts.AB_image_name_label_path
img_name_label_dict={}
with open(AB_image_name_label_path)as file_name_label:
    lines_name_label=file_name_label.readlines()
for k in range(len(lines_name_label)):
    img_name_label_dict.update({lines_name_label[k].strip().split()[0]:lines_name_label[k].strip().split()[1]})

#####extract latents(class-associated codes)
A_img_path=opts.A_img_path
B_img_path=opts.B_img_path
latentAB_save_path=opts.latentAB_save_path

with open(latentAB_save_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # columns_name
    columns_name_list=['image_name']
    for latent_ids in range(style_dim):
        columns_name_list.append(str(latent_ids))
    columns_name_list.append('label')
    writer.writerow(columns_name_list)

    with torch.no_grad():
        ###get class-associated codes from A images
        img_n = 0
        for image_name in os.listdir(A_img_path):
            img_n=img_n+1
            example_path = A_img_path + image_name
            example_img=Variable(transform(Image.open(example_path).convert('RGB')).unsqueeze(0).to(
                device))
            c_A, s = encode(example_img)

            row_list = [image_name]
            for j in range(s.size(1)):
                row_list.append(str(s[0][j].item()))

            row_list.append(img_name_label_dict[image_name])
            writer.writerow(row_list)

            print('A  '+str(img_n))

        ###get class-associated codes from B images
        img_n = 0
        for image_name in os.listdir(B_img_path):
            img_n=img_n+1
            example_path = B_img_path + image_name
            example_img=Variable(transform(Image.open(example_path).convert('RGB')).unsqueeze(0).to(
                device))
            c_A, s = encode(example_img)

            row_list = [image_name]
            for j in range(s.size(1)):
                row_list.append(str(s[0][j].item()))

            row_list.append(img_name_label_dict[image_name])
            writer.writerow(row_list)

            print('B  '+str(img_n))

print("Latents(class-associated codes) extraction finished")


df = pd.read_csv(latentAB_save_path)

feature_names = [c for c in df.columns if c not in ["image_name","label"]]
X = np.array(df[feature_names])
y = np.array(df["label"])

# Create images for a custom tooltip array
tooltip_s = np.array(df["image_name"])

# need to make sure to feed it as a NumPy array, not a list

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(X, projection=sklearn.manifold.TSNE(n_iter=500))

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(7,0.49),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")

# Tooltips with image data for every cluster member
html_name01=opts.html_name01
mapper.visualize(
    graph,
    title="latent Mapper",
    path_html=html_name01,
    custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
html_name02=opts.html_name02
mapper.visualize(
    graph,
    title="latent Mapper",
    path_html=html_name02,
    custom_tooltips=y,
)

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()

