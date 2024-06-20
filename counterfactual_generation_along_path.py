import pandas as pd
import argparse
import numpy as np
import torch
from trainer_exchange import trainer
from torchvision import transforms
import os
from torch.autograd import Variable
from PIL import Image
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--AB_img_path',type=str,default='testAB_img/') ###images involved in the nodes of the topology graph are all placed in this folder
parser.add_argument('--Nodes_codes_center_save_path',type=str,default='Nodes_codes_center.csv')
parser.add_argument('--CAML_trained_gen_model_path',type=str,default='models/gen_00150000.pt')
parser.add_argument('--style_dim',type=int,default=8)
parser.add_argument('--train_is',type=str,default='False')
parser.add_argument('--shortest_path_save_path',type=str,default='AB_images_shortest_path_save.csv')
parser.add_argument('--counterfactual_generation_save_path',type=str,default='counterfactual_generation_results/')
opts = parser.parse_args()

if opts.cuda=='True':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

AB_img_path=opts.AB_img_path

##For each two images(points), their shortest path in the graph were recorded in this file
AB_images_shortest_path=opts.shortest_path_save_path

##For each node in the graph, the center vector(mean values) of all class-association codes involved into this node were recorded in this file
Nodes_codes_center_path=opts.Nodes_codes_center_save_path

##where counterfactual generation results for each two points(images) will be saved
counterfactual_generation_save_path=opts.counterfactual_generation_save_path

if not os.path.exists(counterfactual_generation_save_path):
    os.makedirs(counterfactual_generation_save_path)

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

AB_images_shortest_file=pd.read_csv(AB_images_shortest_path)
Nodes_codes_center_file=pd.read_csv(Nodes_codes_center_path)



with torch.no_grad():
    for i in range(len(AB_images_shortest_file)):
        A_img_name=AB_images_shortest_file.loc[i]['A(start_img)']
        B_img_name=AB_images_shortest_file.loc[i]['B(end_img)']
        A2B_shortest_path=AB_images_shortest_file.loc[i]['A2B_shortest_path']

        counterfactual_generation_save_path2=counterfactual_generation_save_path+A_img_name+'_'+B_img_name+'/'
        if not os.path.exists(counterfactual_generation_save_path2):
            os.makedirs(counterfactual_generation_save_path2)

        examplarA_path = AB_img_path + A_img_name
        examplarA_img = Variable(transform(Image.open(examplarA_path).convert('RGB')).unsqueeze(0).to(device))

        counter_examplarB_path = AB_img_path + B_img_name
        examplarB_img = Variable(transform(Image.open(counter_examplarB_path).convert('RGB')).unsqueeze(0).to(device))

        vutils.save_image((examplarA_img.data + 1) / 2,
                          counterfactual_generation_save_path2 + 'examplar_A_'+A_img_name,
                          padding=0, normalize=False)

        vutils.save_image((examplarB_img.data + 1) / 2,
                          counterfactual_generation_save_path2 + 'counter_examplar_B_' + B_img_name,
                          padding=0, normalize=False)


        c_A_ori, s_A_ori = encode(examplarA_img)


        for j in range(len(A2B_shortest_path.strip().split(','))):
            Node_ID=A2B_shortest_path.strip().split(',')[j]

            data_B_CS = Nodes_codes_center_file.loc[Nodes_codes_center_file['Node_ID'] == Node_ID]
            data_B_CS_numpy = np.array(data_B_CS.values.tolist()[0][1:])

            data_B_CS_tensor = torch.tensor(data_B_CS_numpy).unsqueeze(0).unsqueeze(2).unsqueeze(3)

            couter_examplar_mid=decode(c_A_ori,data_B_CS_tensor)

            vutils.save_image((couter_examplar_mid.data + 1) / 2,
                              counterfactual_generation_save_path2 + 'counter_examplar_mid_N' + str(j)+'.jpg',
                              padding=0, normalize=False)


        print('generation_processing: '+str(i)+'  '+str(j))


















