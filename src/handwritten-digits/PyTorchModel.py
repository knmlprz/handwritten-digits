import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt


def prepare_image(im, output_size=28, blur_kernel_size=7):
    """Prepares an image for training
    
    Parameters
    ----------
        im : PIL image
            An input image.
        output_size : integer, optional
            The size of the output image. Default 28.
        blur_kernel_size : integer, optional
            The size of the Gaussian blur kernel. Default 7.
    """
    # Convert to grayscale PyTorch Tensor
    ptim = T.ToTensor()(T.Grayscale()(im)).squeeze()
    back_val = min(ptim.flatten()).item()
      
    # Find the digit in the image
    top, left = torch.amin(torch.stack(torch.where(ptim > back_val)), dim=1)
    bottom, right = torch.amax(torch.stack(torch.where(ptim > back_val)), dim=1)
    center = torch.tensor([
        (top.item() + bottom.item()) // 2, 
        (left.item() + right.item()) // 2,
    ])
    imrad = max(bottom - center[0], right - center[1])
    
    # Make a padding if the (center + radius) is biger 
    # then an input image or (center - radius) < 0.
    if sum(center + imrad < torch.tensor(ptim.size())) < 2 or sum(center - imrad > 0) < 2:
        
        br_offsite = abs(min(torch.tensor(ptim.size()) - (center + imrad)))
        tl_offsite = abs(min(center - imrad))
        pad = max(br_offsite, tl_offsite)
        ptim = T.Pad(padding=pad.item(), fill=back_val)(ptim)
        center += pad
    
    # Crop, blur, resize and normalize the image
    topc, leftc = center - imrad
    bottomc, rightc = center + imrad
    ptim = ptim[topc:bottomc, leftc:rightc]
    original_size = ptim.size()
    ptim = ptim.view(1, 1, *ptim.size())    # same as .unsqueeze(dim=0) applied twice
    ptim = T.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 7))(ptim)
    ptim = T.Resize(size=[output_size-2, output_size-2])(ptim)
    ptim = T.Pad(padding=1, fill=back_val)(ptim)
    mean, std = ptim.mean(), ptim.std()
    ptim = T.Normalize(mean, std)(ptim)
    orig_size_ptim = T.Resize(size=original_size)(ptim)
    
    return ptim, orig_size_ptim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        ) 
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Classifier:

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN()
        if device == "cuda":
            self.model.cuda()
        self.model.load_state_dict(torch.load(
            "Notebooks/models/model.pth",
            map_location=torch.device(device)
            ))

    def predict(self, image):
        prepim, orig_size_prepim = prepare_image(image)
        plt.cla()
        plt.imshow(prepim[0][0])
        plt.xticks([])
        plt.yticks([])
        plt.show()
        with torch.no_grad():
            output = self.model(prepim)

        predicted_val = output.data.max(1, keepdim=True)[1][0].item()
        return predicted_val, orig_size_prepim
