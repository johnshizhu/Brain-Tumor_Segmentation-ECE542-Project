import sys
sys.path.insert(0, '../')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import matplotlib.patches as mpatches
from unet.model import GeneralUNet
from utils.data_utils import BratsDataset3D
from utils.predict import ModelPredict
import SimpleITK as hh 
import utils.Eval_mri as em
import seaborn as sns



model = GeneralUNet(in_channels=4,  # Adjust based on your dataset's specifics
                    conv_kernel_size=3,
                    pool_kernel_size=2,
                    up_kernel_size=2,
                    dropout=0.1,
                    conv_stride=1,
                    conv_padding=1,
                    conv3d=True,
                    size=4,  # Adjust the number of layers in the UNet
                    complex=4)  # Adjust the complexity or number of initial features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = r'C:\Users\rmahdav2\Desktop\Samaneh\ECE\ECE_500\Brain-Tumor_Segmentation-ECE542-Project-main\Brain-Tumor_Segmentation-ECE542-\Brain-Tumor_Segmentation-ECE542-Project-main\350K_model_dict_correct'
model.load_state_dict(torch.load(model_dir,map_location ='cpu'))
model.to(device)


scan_dir = r'C:\Users\rmahdav2\Documents\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
scan_index = 10
dataset = BratsDataset3D(scan_dir)
model.eval()

[input_tensor, label_tensor] = dataset.__getitem__(scan_index)

# Move input to GPU
input_tensor = input_tensor.to(device)
input_tensor = torch.unsqueeze(input_tensor, 0)

model_output = model(input_tensor)
binary_output = torch.where(model_output < model_output.max()/2, torch.zeros_like(model_output), torch.ones_like(model_output))

scan_np  = input_tensor.squeeze().cpu().numpy()
pred_np  = binary_output.squeeze().detach().cpu().numpy()  
label_np = label_tensor.squeeze().cpu().numpy()  
print(scan_np.shape)
print(pred_np.shape)
print(label_np.shape)


##### metrcs calculation for two labels
em_eval = em.Eval_mri(2)

metrics = em_eval.evaluation_emtrics(label_np, pred_np)
print('Metrics to evaluate the model: ')
print (metrics)


##### Confusion matrix generation
con_matrix = em_eval.generate_matrixConfusion(label_np, pred_np)
con_matrix = con_matrix/np.sum(con_matrix)

tp, tn, fp, fn = con_matrix[0][0], con_matrix[1][1], con_matrix[0][1], con_matrix[1][0]
false_omission_rate = fp/(fp+tp)
accuracy = (tp + tn)/(tp + tn + fp + fn)

plt.figure(figsize = (6,6))
sns.heatmap(con_matrix, cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
plt.xlabel("predictions")
plt.ylabel("real values")
plt.show()




####### Ovelay the TP, TN, FN, FP on the image
alpha = 0.6
confusion_matrix_colors = {
    "tp": (0, 1, 1),  # cyan
    "fp": (1, 0, 1),  # magenta
    "fn": (1, 1, 0),  # yellow
    "tn": (0, 0, 0)  # black
}

middle_slice = 65  # Adjust as needed
current_slice = middle_slice + 3 * 0
scan_slice = scan_np[0, :, :, current_slice] if scan_np.ndim == 4 else scan_np[:, :, current_slice]
validation_mask = em_eval.get_confusion_matrix_overlaid_mask(scan_slice, label_np[:,:,current_slice], pred_np[:,:,current_slice], alpha, confusion_matrix_colors)
print('Cyan - TP')
print('Magenta - FP')
print('Yellow - FN')
print('Black - TN')
plt.imshow(validation_mask)
plt.axis('off')
plt.title('confusion matrix overlay mask')
plt.show()

c = 2
