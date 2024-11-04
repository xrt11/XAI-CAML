# CAML
This is the official code repository for "Bridging the Interpretability Gap for Medical AI Models using Class-Association Manifold Learning".


## Abstract
Explainability has increasingly become a core requirement for intelligent medical device software. Nevertheless, current medical AI technologies suffer from the `interpretability gap' despite tremendous efforts for enhancing explainability. Here we propose class-association manifold learning, a generative AI technology to enhance explanation of black-box medical AI models. The advantage of our method is to represent global knowledge of black-box AI models in a low dimensional mapping, while preserving near-perfect diagnostic decision accuracy, which is not achievable by previous methods. This is made possible by a manifold learning approach that efficiently decouples common decision-related patterns against individual backgrounds. The extracted knowledge is further used to enable AI-generated modifications on arbitrary samples and visualize differential diagnosis rules. Moreover, we develop a topology map to model the entire decision rule set, so that the logic underlying black-box models can be intuitively explicitated by traversing the map and generating virtual contrastive examples. We carry out experiments on an extensive set of medical AI tasks, showing that our method not only achieves much higher accuracy in explaining the behavior of medical AI models, but also helps with extracting medical-compliant knowledge that are unknown during model training, thus providing a potential means of assisting clinical rule and medical knowledge discovery with AI techniques.


## 0. Main Environments
```bash
conda create -n CAML python=3.9
conda activate CAML
pip install torch==1.12.1
pip install torchvision==0.13.1
pip install numpy==1.24.4
pip install pandas==1.3.5
pip install kmapper==2.1.0
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.1
pip install pillow==9.0.1
pip install opencv-python==4.10.0.84
pip install opencv-python-headless==4.10.0.84
pip install openpyxl==3.1.5
pip install networkx==3.1
pip install argparse==1.1
```



## 1. Prepare the dataset
## Original datasets could be downloaded from:
The retinal Optical Coherence Tomography (OCT) and the Chest X-rays image datasets are available at https://data.mendeley.com/datasets/rscbjbr9sj/2.
The Pathologic Myopia Challenge (PALM) dataset can be found at https://ieee-dataport.org/documents/palm-pathologic-myopia-challenge.
The OIA-DDR dataset is available at https://github.com/nkicsl/DDR-dataset.
The Brain Tumor dataset 1 can be downloaded from https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection.
The Brain Tumor dataset 2 can be found at https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.
The Retinal Fundus Multi-Disease Image Dataset (RFMID) is available for download at: https://riadd.grand-challenge.org/download-all-classes/.

## Data used for training and testing in our paper is available at:
The data used for training and testing in our experiment can be downloaded from https://drive.google.com/drive/folders/1qh8fGZ6aqVSRtWpGHtWdnb6vqRDFUbO\\Y?dmr=1\&ec=wgc-drive-hero-goto

## Explanation Regarding Training and Testing Datasets
- After downloading the datasets, you are supposed to put them into './data/', and the file format reference is as follows. (take the Brain Tumor2 dataset as an example.)

- './data/Brain Tumor2/'
  - trainA_img
    - .png
  - trainB_img
    - .png
  
  - testA_img
    - .png
  - testB_img
    - .png

- training images with normal label will be put into 'trainA_img/' folder, while training images with abnormal labels will be put into 'trainB_img/'
- test images with normal label will be put into 'testA_img/' folder, while test images with abnormal labels will be put into 'testB_img/'
- the names and the labels of the training images (with the format 'image_name label') are put into the 'trainAB_img-name_label.txt'
- the names and the labels of the test images (with the format 'image_name label') are put into the 'testAB_img-name_label.txt'
- '0' represents normal class label while other numbers represent abnormal classes in our work

## 2. Train the CAML
```bash
cd code/CAML_Train
python main_train.py  # Train and test CAML model.
```

## 3. Obtain the trained CAML models and generated results of some cases
- After trianing, you could obtain the trained models in './code/CAML_Train/results/models/'
- After trianing, you could obtain the generated results of some cases in './code/CAML_Train/results/images_display/'

## 4. Perform class-associated codes analysis on test datasets
```bash
cd code/CL_Analysis
python CL_codes_extract.py  # Extract class-associated codes of the test images using trained models.
python topological_analysis.py  # Perform topological analysis on extracted class-associated codes.
python tsne_analysis.py  #  Perform t-SNE analysis on extracted class-associated codes.
```

## 5. Perform instance explanation using the class-associated manifold
- After performing topological analysis on extracted class-associated codes, we can get one topological graph, and the images involved into each node within the graph should be recorded (format: ID_number image1_name image2_name...) into the 'Nodes_images_name.txt' file.
```bash
cd code/Case_Show
python graph_nodes_distance_matrix.py  # Calculate the distances between each two nodes within the topological graph obtained from the class-associated codes, and output one distance matrix representing the relations between nodes.
python shortest_path_get_for_each_two_points.py  # For one instance to be explained, we select one counter reference sample for it, and calculate the shortest path between these two sample based on the nodes distance matrix.
python local_explanation_on_instance.py  #  Along the shortest path, we obtain meaningful class-associated codes for guided counterfactual generation, and by analyzing the changes of the generated samples and the changes of the outputs of the black-box model on the generated samples, we can get one saliency map for the instance explanation. 
```
