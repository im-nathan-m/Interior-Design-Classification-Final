import torch
from PIL import Image
from torchvision.transforms import v2
from torch.nn.functional import softmax

from model import ConvNet
# REPLACE "trainModelFile" WITH THE NAME OF THE FILE WITH THE MODEL CLASS

styles = [
    "asian",
    "coastal",
    "contemporary",
    "craftsman",
    "eclectic",
    "farmhouse",
    "french-country",
    "industrial",
    "mediterranean",
    "mid-century-modern",
    "modern",
    "rustic",
    "scandinavian",
    "shabby-chic-style",
    "southwestern",
    "traditional",
    "transitional",
    "tropical",
    "victorian"
]

# create the model class, and load the weights. make sure "model.pt" matches
# the filename you used when saving the model (should be in the same folder as this file)
model = ConvNet()
model.load_state_dict(torch.load("model-v1.pt", weights_only=True, map_location=torch.device('cpu')))

# set to eval mode (only matters if you are using dropout)
model.eval()

# transforms are only for resizing the image or necessary other commands
# make sure resize pixels here match your model, replace (100,100) with your size!
test_transforms = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToTensor()
])

# load the file "image.png", change this to your file name
# demo.jpg is rustic_111.jpg from testing dataset
# demo1.jpg is shabby-chic-style_75.jpg from testing dataset
img = Image.open("demo/demo.jpg").convert('RGB')
# apply transformations (resizing) to the image
img = test_transforms(img)

# print(img.shape) # check image shape is correct, if it isn't, unsqueeze
img = torch.unsqueeze(img, 0)

pred = model(img)

probs = softmax(pred, dim=1)
max_prob, pred_class = torch.max(probs, dim=1)

print(f"Probabilities: {probs}")
print(f"Predicted class: {styles[pred_class.item()]}, confidence: {max_prob.item()}")
# at minimum the output should print a prediction, but if you are doing classification,
# use Softmax to turn the output into percentages 
# (see week 4 day 2 activity document on canvas)
# also, try to convert the raw number output into understandable classes