import torch
import cv2
from torchvision.transforms import Compose
from utils.transforms import Resize, NormalizeImage, PrepareForNet
from datasets.load_image_and_depth import LoadImageDepthAndLabels
from models.doepd_net import DoepdNet
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

def train(batch_size, epochs):
    net_w, net_h = 384, 384
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print("device: %s" % device)
    image_transforms = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    train_data = LoadImageDepthAndLabels(train=True, transforms= image_transforms)
    # test_data = LoadImageDepthAndLabels(train=False, transforms= image_transforms)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = batch_size, shuffle = True, **kwargs
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size = batch_size, shuffle = True, **kwargs
    # )
    
    model = DoepdNet(run_mode='midas').to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    loss = nn.MSELoss(reduction="mean")   
    pbar = tqdm(train_loader, position = 0, leave = True)
    for epoch in range(epochs):
        for batch_idx, (image, target_depth) in enumerate(pbar):
            optimizer.zero_grad()
            image, target_depth = image.float().to(device), target_depth.float().to(device)
            image = image / 255.0
            midas_pred = model(image)[1]
            loss(midas_pred, target_depth)
            loss.backward()
            optimizer.step()
            pbar.set_description(desc= f'Epoch = {epoch} Loss={loss.item()} Batch_id={batch_idx}')
        chkpt = {
            "epoch": epoch,
            "model": model.state_dict()
        }
        torch.save(chkpt, "/content/drive/MyDrive/doepd/weights/midas_last.pt")

    
if __name__=='__main__':
    BATCH_SIZE = 64
    # WEIGHT_PATH = "weights/midas_v21-f6b98070.pt"
    EPOCHS = 50
    train(batch_size = BATCH_SIZE, epochs = EPOCHS)