from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import os

class FasterRCNN_WoodDataset(Dataset):
    def __init__(self, image_dir, coco, transforms=None) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.coco = coco
        self.transforms = transforms

    def __getitem__(self, index:int):
        index += 1
        image_id = index
        image_info = self.coco.imgs[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds([index]))
        n = len(anns)

        image = cv2.imread(os.path.join(self.image_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        boxes=np.zeros((n, 4))
        area = np.zeros(n)
        iscrowd = np.zeros(n)
        labels = np.zeros(n, dtype = np.int64)
        image_ids = np.zeros(n)

        for i in range(n):
          boxes[i, :] = anns[i]['bbox']
          area[i] = anns[i]['area']
          iscrowd[i] = anns[i]['iscrowd']
          labels[i] = anns[i]['category_id']
          image_ids[i] = anns[i]['image_id']
        boxes_yolo = boxes.copy()
        boxes_yolo[:, 2] = boxes[:,0] + boxes[:,2]
        boxes_yolo[:, 3] = boxes[:,1] + boxes[:,3]

        check = area > 10
        area = area[check]
        boxes_yolo = boxes_yolo[check]
        labels = labels[check]
        iscrowd = iscrowd[check]
        image_ids = torch.tensor([index])

        boxes = torch.as_tensor(boxes_yolo, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_ids = torch.as_tensor(image_ids, dtype=torch.int64)

        target={}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_ids
        target['area'] = area
        target['iscrowd'] = iscrowd
            
        if self.transforms:
                sample = {
                    'image': image, 'bboxes': target['boxes'], 'labels': target['labels']
                }
                sample = self.transforms(**sample)
                image = sample['image']
                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                target['boxes'] = target['boxes'].type(torch.float32)
        return  image, target
        
    def __len__(self) -> int:
        return len(self.coco.imgs)