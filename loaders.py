import numpy as np
from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import torchvision.transforms.functional as TF

def create_mask(data):
    mask = []

    # Attack
    mask.append(1 if len(np.intersect1d(data, ["AttackSpeed","CriticalStrike","LifeSteal","SpellDamage","SpellDamage","SpellVamp","Damage",])) > 0 else 0) 
    # Defense
    mask.append(1 if len(np.intersect1d(data, ["Health","Armor","SpellBlock","HealthRegen",])) > 0 else 0) 
    # Physical
    mask.append(1 if len(np.intersect1d(data, ["AttackSpeed","CriticalStrike","LifeSteal","Armor","Damage",])) > 0 else 0) 
    # Magic
    mask.append(1 if len(np.intersect1d(data, ["AbilityHaste","CooldownReduction","SpellDamage","ManaRegen","SpellDamage","Mana","SpellBlock","SpellVamp",])) > 0 else 0) 
    # Utility
    mask.append(1 if len(np.intersect1d(data, ["AbilityHaste","CooldownReduction","Vision","Stealth","Consumable","AttackSpeed","ManaRegen","Boots","Health","Mana","LifeSteal","SpellVamp","HealthRegen",])) > 0 else 0) 
    
    return mask

class ItemDataset(Dataset):
    def __init__(self, json_path, imgs_path, img_size, transform=None):
        self.jsonPath = json_path
        self.transform = transform
        self.img_size = img_size
        f = open(json_path, encoding="utf8")
        self.jsonData = json.load(f)['data']
        self.items = []
        self.tag_masks = []
        for itemKey in self.jsonData.keys():
            data = self.jsonData[itemKey]
            img_path = f"{imgs_path}/{itemKey}.png"
            self.items.append(img_path)
            self.tag_masks.append(create_mask(data['tags']))

    def __len__(self):
        return len(self.jsonData)
    
    def __getitem__(self, idx):
        img_path = self.items[idx]
        mask = self.tag_masks[idx]

        image = Image.open(img_path).resize([self.img_size, self.img_size])
        x = TF.to_tensor(image)
        sample = {
            "image": x,
            "mask": torch.from_numpy(np.array(mask))
        }
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['combined'] = torch.cat((sample['image'].flatten(), sample['mask']),-1)
        return sample