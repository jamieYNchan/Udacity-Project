import argparse
import Common as common
import importlib
import time
importlib.reload(common)
parser = argparse.ArgumentParser(description= 'Welcome to Image Classifier Training part.')

# python train.py data_dir --save_dir save_directory
parser.add_argument('data_dir', action="store", default="./flowers/", 
                    help = 'Enter path of folder which msuh included train, test and valid folder.')

parser.add_argument('--save_dir',type=str, dest="save_dir", action="store", default="checkpoint.pth",
                    help=" Save Directory")
#parser.add_argument('save_directory', dest="save_directory", action="store_true", default=False,
#                    help=" Save checkpoints")

# python train.py data_dir --arch "vgg13"
parser.add_argument('--arch', type=str, action="store", dest = "pretrained_model", default = "densenet121",
                    help= "Only 'densenet121' and 'vgg13' are supported in this model. Default densenet121.")

# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.003, help="Input custom learning rate. Default:0.01")
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
parser.add_argument('--dropout', type=float, dest = "dropout", action = "store", default = 0.3)
parser.add_argument('--epochs', type=int, dest="epochs", action="store", default=3)

# python train.py data_dir --gpu
parser.add_argument('--gpu', action="store_true", default=False,
                    help="Enable GPU mode. Default off.")

read_line = parser.parse_args()
data_dir = read_line.data_dir
save_dir = read_line.save_dir
structure = read_line.pretrained_model
lr = read_line.learning_rate
hidden_units = read_line.hidden_units
dropout = read_line.dropout
epochs = read_line.epochs
gpu = read_line.gpu
if __name__ == "__main__":
    print("Data Dir:", data_dir)
    print("save_dir:", save_dir)
    if not read_line.data_dir is None:
        common.init(gpu)
    # Start Training
    start_time = time.time()
    data_transforms, image_datasets, dataloaders = common.load_data(data_dir)
    common.create_model(structure=structure, dropout=dropout, hidden_units= hidden_units, lr=lr)
    common.train(epochs)
    common.save_checkpoint(save_dir,structure, hidden_units, dropout, lr, epochs)
    elapsed_time = time.time() - start_time
    
    print(f'Total Traning Time: {"%.2f" %elapsed_time}s')
