###paths setting###
path_train = '../gt_jsons/train_hls.json'
path_val = '../gt_jsons/val_hls.json'
path_test = '../gt_jsons/test_hls.json'
root_dataset_path = '/hls_data_mix'
#root_dataset_path = '/thunen_pretrain_area_3/'

kfold_labels_path = 'labels_all_thesis_parcels.json'
split_list = ['train','validation']
###batch size###
train_batch = 512
val_batch = 512
test_batch = 256
num_workers_train = 0
num_workers_val = 0
num_workers_test = 0
###mlp size settings###
mlp1_size=[5, 16, 32, 64]
input_channels = 1
input_channels = 1
###train settings###
number_of_epochs = 500
save_dir = "finetuned_models"
embed_dim = 64
dropout_val = 0.1
size_of_mlp_tr = 1024
tr_layers = 4
n_heads = 3
number_of_classes = 5
learning_rate = 1e-3
number_of_layers = 4
###MLP multivariate version###
in_size = 240
hid1 = 512




path_to_pretrained_resnet = './epoch=17-step=1314.ckpt'
path_to_pretrained_transformer = './epoch=37-step=2774.ckpt'