from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Pix2PixDataset(Dataset):
    def __init__(self, files):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        self.len_ = len(self.files)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.Resize(size=(286, 286 * 2)),
            transforms.Resize(size=(256, 256 * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = self.load_sample(self.files[index])
        x = transform(x)

        w = x.shape[2] // 2
        real_image = x[:, :, w:]
        edges_image = x[:, :, :w]

        return real_image, edges_image