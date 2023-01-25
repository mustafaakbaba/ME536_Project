import torch
import cv2
import torchvision.transforms as transforms
import argparse
from detection_utils import draw_bboxes


# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define the image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# initialize and set the model and utilities
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
ssd_model.to(device)
ssd_model.eval()
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

# read the image
image_path = "/home/betul/Documents/my_project/data/1.png"
image = cv2.imread(image_path)
# keep the original height and width for resizing of bounding boxes
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# apply the image transforms
transformed_image = transform(image)
# convert to torch tensor
tensor = torch.tensor(transformed_image, dtype=torch.float32)
# add a batch dimension
tensor = tensor.unsqueeze(0).to(device)

# get the detection results
with torch.no_grad():
    detections = ssd_model(tensor)
# the PyTorch SSD `utils` help get the detection for each input if...
# ... there are more than one image in a batch
# for us there is only one image per batch
results_per_input = utils.decode_results(detections)
# get all the results where detection threshold scores are >= 0.45
# SSD `utils` help us here as well
best_results_per_input = [utils.pick_best(results, 0.45) for results in results_per_input]
# get the COCO object dictionary, again using `utils`
classes_to_labels = utils.get_coco_object_dictionary()
image_result = draw_bboxes(image, best_results_per_input, classes_to_labels)
cv2.imshow('Detections', image_result)
cv2.waitKey(0)
# save the image to disk

cv2.imwrite(f"outputs/{save_name}", image_result)
