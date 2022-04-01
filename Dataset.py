import torch
import os, glob
import random, csv
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# args = parse_args()
model_select = 'ResNet10'
model_select = 'Conv6'

class SOURCE_DATA(Dataset):

    def __init__(self, setname):
        super(SOURCE_DATA, self).__init__()
        self.root = setname
        if model_select == 'Conv4' or model_select == 'Conv6':
            self.resize = 84
        else:
            self.resize = 224
        self.name2label = {}

        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())


        self.data, self.label = self.load_csv('source.csv')

        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.tif'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'
            print(len(images), images)

            #random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)


        assert len(images) == len(labels)

        return images, labels
    def denormalize(self, x_hat):

        mean = [0.485, 0.456,0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x


class TARGET_DATA(Dataset):

    def __init__(self, setname):
        super(TARGET_DATA, self).__init__()
        self.root = setname
        if model_select == 'Conv4' or model_select == 'Conv6':
            self.resize = 84
        else:
            self.resize = 224
        self.name2label = {}

        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())


        self.data, self.label = self.load_csv('target.csv')

        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.tif'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'
            print(len(images), images)

            #random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)


        assert len(images) == len(labels)

        return images, labels
    def denormalize(self, x_hat):

        mean = [0.485, 0.456,0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x
class QUERY_DATA(Dataset):

    def __init__(self,csv_path):
        super(QUERY_DATA, self).__init__()
        if model_select == 'Conv4' or model_select == 'Conv6':
            self.resize = 84
        else:
            self.resize = 224
        self.name2label = {}

        self.data, self.label = self.load_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(filename)):
            print('query csv not exist')
        # read from csv file
        images, labels = [], []
        with open(os.path.join(filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)


        assert len(images) == len(labels)

        return images, labels
    def denormalize(self, x_hat):

        mean = [0.485, 0.456,0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x

def get_shot_data(shot_path):
    # read from csv file
    image_path, labels = [], []
    with open(os.path.join(shot_path)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)

            image_path.append(img)
            labels.append(label)

    assert len(image_path) == len(labels)

    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    images = []
    for i in range(0,len(labels)):
        path = image_path[i]
        image = transform(Image.open(path).convert('RGB'))
        images.append(image)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels