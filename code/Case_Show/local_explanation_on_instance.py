import pandas as pd
import argparse
import heapq
import numpy as np
import csv
import torch
from torch.autograd import Variable
from PIL import Image
from trainer_exchange import trainer
from torchvision import transforms
import os
import torchvision.utils as vutils
from torch.nn import functional
import cv2
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--AB_img_path',type=str,default='/data/Brain Tumor2/testAB_img/') ###images involved in the nodes of the topology graph are all placed in this folder
parser.add_argument('--Nodes_codes_center_save_path',type=str,default='/code/Case_Show/results/Nodes_codes_center.csv')
parser.add_argument('--shortest_path_save_path',type=str,default='/code/Case_Show/results/AB_images_shortest_path_save.csv')
parser.add_argument('--CAML_trained_gen_model_path',type=str,default='/code/trained_models/CAML_brain_trained_model.pt')
parser.add_argument('--outer_classifier_path',type=str,default='/code/trained_models/blackbox_classifier(res50)_brain_trained_model.pth') #### black-box model whose behaviors on local instance will be explained
parser.add_argument('--style_dim',type=int,default=8)
parser.add_argument('--train_is',type=str,default='False')
parser.add_argument('--generate_img_save_path',type=str,default='/results/generate_img_saliency_maps_for_local_explanation/')



explained_case_name='z_86_01567.png'
counter_reference_case_name='z_86_01318.png'


# function of saliency map generation based the difference map
def heatmap_show(image,difference_map):
    image = image.data.cpu().numpy()
    saliency_map = difference_map.data.cpu().numpy()
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0, 1)
    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    image = np.uint8(image * 255).transpose(1, 2, 0)
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap01 = img_with_heatmap / np.max(img_with_heatmap)
    img_with_heatmap=np.uint8(255 * img_with_heatmap01)
    return color_heatmap, img_with_heatmap


opts = parser.parse_args()


AB_img_path=opts.AB_img_path
generate_img_save_path=opts.generate_img_save_path


####data preprocessing
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.CenterCrop((256, 256))] + transform_list
transform_list = [transforms.Resize(256)] + transform_list
transform = transforms.Compose(transform_list)

Nodes_codes_center_save_path=opts.Nodes_codes_center_save_path
Nodes_codes_center_file=pd.read_csv(Nodes_codes_center_save_path)

shortest_path_save_path=opts.shortest_path_save_path
shortest_path_file=pd.read_csv(shortest_path_save_path)

try:
    shortest_path_nodes=shortest_path_file.loc[(shortest_path_file['A(start_img)'] == explained_case_name) & (shortest_path_file['B(end_img)'] == counter_reference_case_name)]['A2B_shortest_path']
    shortest_path_nodes_list=shortest_path_nodes.values.tolist()[0].strip().split(',')

except:
    print('Path from ' + explained_case_name + ' to ' + counter_reference_case_name + ' is not found in the ' + shortest_path_save_path)
    sys.exit(1)


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

if not os.path.exists(generate_img_save_path):
    os.makedirs(generate_img_save_path)

explained_case_path = AB_img_path + explained_case_name
example_img=Variable(transform(Image.open(explained_case_path).convert('RGB')).unsqueeze(0).to(device))
I_A, s = encode(example_img)

outer_classifier_path=opts.outer_classifier_path
classify_model=torch.load(outer_classifier_path,map_location=device)
classify_model=classify_model.eval()
classify_model=classify_model.to(device)

pre_img_a = classify_model(example_img)
pre_img_a = functional.softmax(pre_img_a, dim=1)
pre_img_a_out_max, pre_img_a_out_max_index = pre_img_a.max(dim=1)
print('example image is predicted as: ' + str(pre_img_a_out_max_index.item()))

for i in range(len(shortest_path_nodes_list)):
    data_A2B_CS=Nodes_codes_center_file.loc[Nodes_codes_center_file['Node_ID'] == shortest_path_nodes_list[i]].values.tolist()[0][1:9]
    data_A2B_CS_tensor=torch.tensor(data_A2B_CS).unsqueeze(0).to(device)
    generate_img_tensor=decode(I_A,data_A2B_CS_tensor)

    vutils.save_image((generate_img_tensor.data + 1) / 2,generate_img_save_path+'ex_'+explained_case_name+'_ref_'+counter_reference_case_name+'_ID_'+str(i)+'_gen.png',
                      padding=0, normalize=False)

    pre_img_sbca = classify_model(generate_img_tensor)
    pre_img_sbca = functional.softmax(pre_img_sbca, dim=1)
    pre_img_sbca_out_max, pre_img_sbca_out_max_index = pre_img_sbca.max(dim=1)
    print('generative image with ' + str(i) + 'd distance from the example')

    if pre_img_sbca_out_max_index.item() != pre_img_a_out_max_index.item() or i==(len(shortest_path_nodes_list)-1):
        # compute the difference between the explained example and the generative counter example
        difference_casb_ca = torch.abs(((generate_img_tensor + 1) / 2) - ((example_img + 1) / 2))

        # generate the saliency map/heatmap based the difference between the explained example and the generative counter example
        colormap, colormap_with_img = heatmap_show((example_img.squeeze(0) + 1) / 2, difference_casb_ca.squeeze(0))

        # save the saliency map
        cv2.imwrite(generate_img_save_path + 'saliency_map_with_img_' + str(i) + 'd.jpg', colormap_with_img)
        print('Continuously changing generative examples with the final saliency map for local explanation is presented in ' + generate_img_save_path)
        break






