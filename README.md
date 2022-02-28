## RAIDS
Security is of primary importance to vehicles. RAIDS employs a lightweight machine learning model to extract road contexts from sensory information (e.g., camera
images and distance sensor values) that are used to generate control signals for maneuvering the car. With such ongoing road context, RAIDS validates corresponding frames
observed on the in-vehicle network. Anomalous frames that substantially deviate from the road context will be discerned
as intrusions. We have implemented a prototype of RAIDS with neural networks, and conducted experiments on a Raspberry Pi with extensive datasets and meaningful intrusion
cases.

## Publication
Jingxuan Jiang, Chundong Wang, Sudipta Chattopadhyay, and Wei Zhang. *Road Context-aware Intrusion Detection System for Autonomous Cars*. In Proceedings of the 21st International Conference on Information and Communications Security (ICICS 2019). Beijing, China. 15-17 December 2019.

Paper link: <https://asset-group.github.io/papers/ICICS19-RAIDS.pdf>

## Environment setting up.

For AWS Server/Ubuntu 
Install python libraries in requirement.txt
Use of virtual environment is recommended.
```
python3 -m venv raids
source ~/raids/bin/activate
pip -r requirements.txt
```
Note 'tensorflow_aarch64==2.7.0' is for ARM processors. Skip is you are using amd64 architecture. 
  
 
## Dataset:
   
  1. Udacity Inc. The Udacity open source self-driving car project, April 2018. <https://github.com/udacity/self-driving-car>
  2. Udacity Inc. Udacity’s self-driving car simulator, July 2017. <https://github.com/udacity/self-driving-car-sim>
  3. Apollo.auto. Roadhackers platform in Baidu Apollo project, April 2018. <http://data.apollo.auto/static/pdf/road_hackers_en.pdf>
  4. Comma.ai. The Comma.ai driving dataset, October 2016. <https://github.com/commaai/research>
  5. Sully Chen. Sully Chen’s driving datasets (2017 & 2018), April 2018. <https://github.com/SullyChen/driving-datasets>
    
## Quick Start
For example as chen_old:
1. First we need to train a CNN model to extract feature from each image. we use CAN information(steering angle) as label to train the models.The dataset is divided into training and testing with the training part being 70%.
'''
python3 dataset/chen_old/data_to_csv.py
'''
2. Preprocess training data. Converting to HSV and finding the difference between 2 images.
'''
python3 chen_data_try/feature_extract_2cnn/preprocess_train_data.py
'''
3. In feature_extraction_2cnn file, use python train.py to train the model and save the best model.
'''
python3 chen_data_try/feature_extract_2cnn/train.py
'''
4. Run abrupt and directed intrusion attacks on the dataset. ~Then try to use intrusion CAN model code modifies some of the values in the csv dataset and also adds a column to it specifying if is an attack or not (1=attack,0=no attack) In try_commai_dark_data_2cnn/attack_csv file, use python attack_model_abs_a_random.py to attack CAN.~
'''
python3 chen_data_try/try_chen_old_data/attack_csv/attack_model_abrupt.py
python3 chen_data_try/try_chen_old_data/attack_csv/chen_old_all_directed_intrusion.csv
'''
5. Train and test intrusion detection with context. The dataset is divided into training and testing with the training part being 70%.
'''
python3 chen_data_try/try_chen_old_data/feature_extraction_intrusion_detection.py
'''
6. Results are saved in 
'chen_data_try/try_chen_old_data/accuracy_file_f_e_predict_a_consecutive.csv'
