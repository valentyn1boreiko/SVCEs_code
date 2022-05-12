import torch
from robustness.tools.imagenet_helpers import ImageNetHierarchy
import random
import numpy as np
import argparse

from utils.datasets.paths import get_imagenet_path
from utils.datasets.imagenet import get_imagenet_labels
import utils.datasets as dl
from utils.visual_counterfactual_generation import targeted_translations
import re


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--gpu','--list', nargs='+', default=[0],
                    help='GPU indices, if more than 1 parallel modules will be called')
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--samples', type=int, default=5)

hps = parser.parse_args()


if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
    num_devices = 1
elif len(hps.gpu)==1:
    device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:' + str(hps.gpu[0]))
    num_devices = 1
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))
    num_devices = len(device_ids)
hps.num_devices = num_devices
ds_path = get_imagenet_path()
ds_info_path = '/mnt/SHARED/datasets/imagenet'

min_descendants = 3
max_descendants = 10

img_size = 224
imgs_per_class = hps.samples

selected_wnids = ['geological formation, formation', 'foodstuff, food product', 'finch', 'edible fruit', 'memorial, monument'] #, 'dish', 'cruciferous vegetable', 'colubrid snake, colubrid', 'beetle', 'amphibian']
#selected_wnids += ['headdress, headgear']
#selected_wnids += ['boat', 'bony fish', 'memorial, monument']
#selected_wnids = ['boat', 'dish']
selected_wnids = None


if selected_wnids is None:
    dir = 'ImageNetHierachyTransfer'
    wnid_found = {}
else:
    dir = 'ImageNetHierachyTransfer/selected_calibrated'
    wnid_found = {wnid: False for wnid in selected_wnids}

dataset = 'imagenet'
bs = hps.bs * len(device_ids)
model_descriptions = [
    ('PytorchResNet50', 'l2_improved_3_ep', 'best', 0.7155761122703552, False)
]

in_hier = ImageNetHierarchy(ds_path, ds_info_path)
in_labels = get_imagenet_labels()
in_loader = dl.get_ImageNet(train=False, augm_type='none', size=img_size)
in_dataset = in_loader.dataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

accepted_wnids = []

for wnid, wnid_in_descendants, _ in in_hier.wnid_sorted:
    if (wnid_in_descendants >= min_descendants) and (wnid_in_descendants <= max_descendants):
        accepted_wnids.append(wnid)

included_imagenet_wnids = set()

pruned_wnids = []

for wnid in accepted_wnids:
    wnid_subclasses = in_hier.get_subclasses([wnid], balanced=False)[0][0]
    intersection_included = included_imagenet_wnids.intersection(wnid_subclasses)
    if len(intersection_included) == 0:
        pruned_wnids.append(wnid)
        included_imagenet_wnids.update(wnid_subclasses)

in_targets = torch.LongTensor(in_dataset.targets)

imgs = torch.zeros((0, 3, 224, 224))
targets_lists = []
filenames = []

for wnid in pruned_wnids:
    wnid_subclasses = in_hier.get_subclasses([wnid], balanced=False)[0][0]

    num_wnid_images = len(wnid_subclasses) * imgs_per_class
    wnid_images = torch.zeros((num_wnid_images, 3, img_size, img_size))
    wnid_images_idx = 0

    wnid_targets_lists = []
    wnid_word = in_hier.wnid_to_name[wnid]
    if wnid_word in wnid_found.keys():
        wnid_found[wnid_word] = True

    wnid_word_subbed = re.sub(r'[^\w\-_\. ]', '', wnid_word )

    for class_idx in wnid_subclasses:
        in_matching_class_idcs = torch.nonzero(in_targets == class_idx, as_tuple=False).squeeze()
        random_selected_idcs = in_matching_class_idcs[torch.randperm(len(in_matching_class_idcs))][:min(imgs_per_class, len(in_matching_class_idcs))]
        subclass_word = re.sub(r'[^\w\-_\. ]', '', in_labels[class_idx] )

        if selected_wnids is None or wnid_word in selected_wnids:
            for i, img_idx in enumerate(random_selected_idcs):
                wnid_images[wnid_images_idx] = in_dataset[img_idx][0]
                wnid_images_idx += 1

                targets_lists.append( list(wnid_subclasses) )

                filename = f'{wnid_word_subbed}_{subclass_word}_{i}_{img_idx}'
                filenames.append(filename)

    if wnid_images_idx > 0:
     imgs = torch.cat([imgs, wnid_images[:wnid_images_idx]], dim=0)

print(f'Number of images {len(imgs)}')
if selected_wnids is not None:
    for wnid in selected_wnids:
        if not wnid_found[wnid]:
            print(f'Not found: {wnid}')

norm = 'L1.5'
if norm.lower() == 'l1':
    radii = [400, 600, 800]
    targeted_translations(model_descriptions, radii, imgs, targets_lists, bs, in_labels, device, dir, dataset,
                                          norm='L1', steps=75, attack_type='apgd', filenames=filenames, device_ids=device_ids)
elif norm.lower() == 'l1.5':
    radii = [50, 75, 100]
    print('num images', len(imgs), len(targets_lists), bs)
    targeted_translations(model_descriptions, radii, imgs, targets_lists, bs, in_labels, device, dir, dataset,
                          norm='l1.5', steps=75, attack_type='afw', filenames=filenames, device_ids=device_ids)
elif norm.lower() == 'l2':
    radii = [12, 18, 24]
    targeted_translations(model_descriptions, radii, imgs, targets_lists, bs, in_labels, device, dir, dataset,
                          norm='L2', steps=75, attack_type='apgd', filenames=filenames, device_ids=device_ids)


