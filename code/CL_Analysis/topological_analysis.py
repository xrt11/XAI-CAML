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
parser.add_argument('--latentAB_save_path',type=str,default='/code/CL_Analysis/results/testAB_CL_codes_extraction_results.csv')  ###where the class-associated codes extracted from the images and the images label are recorded
parser.add_argument('--html_name01',type=str,default='/results/topological_analysis_custom_image_name_result.html')  ###topological analysis results
parser.add_argument('--html_name02',type=str,default='/results/topological_analysis_custom_labels_result.html') ###topological analysis results


opts = parser.parse_args()


latentAB_save_path=opts.latentAB_save_path

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

