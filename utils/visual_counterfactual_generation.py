import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os
import pathlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.load_trained_model import load_model
from tqdm import trange
from time import sleep
from .train_types.helpers import create_attack_config, get_adversarial_attack
from PIL import Image

def _prepare_targeted_translations(model_descriptions, imgs, target_list,
                                   class_labels, device, dataset, bs, device_ids=None):
    num_classes = len(class_labels)
    num_datapoints = len(imgs)
    num_models = len(model_descriptions)
    probabilities = torch.zeros((num_datapoints, len(model_descriptions), num_classes), dtype=torch.float32)

    for model_idx in trange(num_models, desc='Models progress'):
        type, folder, checkpoint, temperature, temp = model_descriptions[model_idx]
        model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
        # search failure cases
        if device_ids is not None and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()

        with torch.no_grad():
            n_batches = int(np.ceil(num_datapoints / bs))
            for batch_idx in range(n_batches):
                datapoint_idx = batch_idx*bs
                datapoint_end_idx = min((batch_idx+1)*bs, num_datapoints)
                data = imgs[datapoint_idx:datapoint_end_idx]
                data = data.to(device)
                orig_out = model(data)
                orig_confidences = torch.softmax(orig_out, dim=1)
                probabilities[datapoint_idx:datapoint_end_idx, model_idx] = orig_confidences.detach().cpu()

    all_models_original_probabilities = probabilities

    max_num_targets = max([len(T) for T in target_list])
    perturbation_targets = torch.empty((num_datapoints, max_num_targets), dtype=torch.long).fill_(-1)
    for i in range(num_datapoints):
        datapoint_target_list = target_list[i]
        datapoint_target_vector = torch.empty(max_num_targets).fill_(-1)
        for j, val in enumerate(datapoint_target_list):
            datapoint_target_vector[j] = val
        perturbation_targets[i] = datapoint_target_vector

    all_models_perturbation_targets = torch.zeros((num_datapoints, num_models, max_num_targets), dtype=torch.long)
    for i in range(num_models):
        all_models_perturbation_targets[:, i, :] = perturbation_targets

    return all_models_perturbation_targets, all_models_original_probabilities


def _find_wrong_examples(model_descriptions, dataloader, num_examples, class_labels, device, dataset, device_ids=None):
    num_classes = len(class_labels)
    num_datapoints = len(dataloader.dataset)
    num_models = len(model_descriptions)
    batch_shape = next(iter(dataloader))[0].shape

    img_dimensions = batch_shape[1:]
    imgs = torch.zeros((num_datapoints, ) + img_dimensions, dtype=torch.float)

    targets = torch.zeros(num_datapoints, dtype=torch.long)
    predictions = torch.zeros((num_datapoints, len(model_descriptions)), dtype=torch.long)
    probabilities = torch.zeros((num_datapoints, len(model_descriptions), num_classes), dtype=torch.float32)
    failure = torch.zeros((num_datapoints, len(model_descriptions)), dtype=torch.bool)

    with torch.no_grad():
        for model_idx in trange(num_models, desc='Models progress'):
            type, folder, checkpoint, temperature, temp = model_descriptions[model_idx]
            model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            # search failure cases
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()

            datapoint_idx = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    if model_idx == 0:
                        imgs[datapoint_idx:(datapoint_idx+data.shape[0]),:] = data
                        targets[datapoint_idx:(datapoint_idx + data.shape[0])] = target

                    data, target = data.to(device), target.to(device)

                    orig_out = model(data)
                    orig_confidences = torch.softmax(orig_out, dim=1)
                    _, orig_pred = torch.max(orig_confidences, 1)
                    correct = orig_pred.eq(target)

                    predictions[datapoint_idx:(datapoint_idx+data.shape[0]), model_idx] = orig_pred.detach().cpu()
                    probabilities[datapoint_idx:(datapoint_idx+data.shape[0]), model_idx] = orig_confidences.detach().cpu()
                    failure[datapoint_idx:(datapoint_idx+data.shape[0]), model_idx] = ~correct

                    datapoint_idx += data.shape[0]

        all_model_failure = torch.sum(failure, dim=1) >= len(model_descriptions)
        all_model_failure_idcs = torch.nonzero(all_model_failure, as_tuple=False).squeeze()

        examples_found = min(all_model_failure_idcs.shape[0], num_examples)
        all_model_failure_idcs = all_model_failure_idcs[:examples_found]

        imgs = imgs[all_model_failure_idcs,:]
        all_models_targets = targets[all_model_failure_idcs]
        all_models_original_probabilities = probabilities[all_model_failure_idcs,:]

        perturbation_targets = torch.zeros((examples_found, num_models, 2), dtype=torch.long)
        for i in range(num_models):
            perturbation_targets[:, i, 0] = all_models_targets
            perturbation_targets[:, i, 1] = predictions[all_model_failure_idcs, i]

        print(f'Found {examples_found} out of {num_examples} falsely classified images')
        return imgs, perturbation_targets, all_models_original_probabilities

def _plot_diff_image(a,b, filepath):
    diff = (a - b).sum(2)
    min_diff_pixels = diff.min()
    max_diff_pixels = diff.max()
    min_diff_pixels = -max(abs(min_diff_pixels), max_diff_pixels)
    max_diff_pixels = -min_diff_pixels
    diff_scaled = (diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)
    cm = plt.get_cmap('seismic')
    colored_image = cm(diff_scaled.numpy())
    pil_img = Image.fromarray(np.uint8(colored_image * 255.))
    pil_img.save(filepath)

def _plot_single_img(torch_img, filepath):
    pil_img = Image.fromarray(np.uint8(torch_img.numpy() * 255.))
    pil_img.save(filepath)

def _plot_counterfactuals(dir, model_name, model_checkpoint, original_imgs, original_probabilities, targets,
                          perturbed_imgs, perturbed_probabilities, radii, class_labels, filenames=None,
                          plot_single_images=False, show_distances=False):


    num_imgs = targets.shape[0]
    num_radii = len(radii)
    scale_factor = 1.5
    for img_idx in trange(num_imgs, desc=f'{model_name} {model_checkpoint} - Image write'):
        if filenames is None:
            single_img_dir = os.path.join(dir, f'{img_idx}')
        else:
            single_img_dir = os.path.join(dir, os.path.splitext(filenames[img_idx])[0])

        if plot_single_images:
            pathlib.Path(single_img_dir).mkdir(parents=True, exist_ok=True)

        img_targets = targets[img_idx,:]
        valid_target_idcs = torch.nonzero(img_targets != -1, as_tuple=False).squeeze(dim=1)
        num_targets =  len(valid_target_idcs)

        num_rows = num_targets
        num_cols = num_radii + 1
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
        if num_rows == 1:
            ax = np.expand_dims(ax, 0)
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        img_probabilities = original_probabilities[img_idx, :]
        img_confidence, img_prediction = torch.max(img_probabilities, dim=0)

        if plot_single_images:
            _plot_single_img(img_original, os.path.join(single_img_dir, 'original.png'))

        if num_targets == 2:
            title = f'{class_labels[img_targets[0]]}: {img_probabilities[img_targets[0]]:.2f}\n' \
                    f'{class_labels[img_targets[1]]}: {img_probabilities[img_targets[1]]:.2f}'
        else:
            title = f'{class_labels[img_prediction]}: {img_confidence:.2f}'

        # plot original:
        ax[0, 0].axis('off')
        ax[0, 0].set_title(title)
        ax[0, 0].imshow(img_original, interpolation='lanczos')

        for j in range(1, num_rows):
            ax[j, 0].axis('off')

        #plot counterfactuals
        for target_idx_idx in range(num_targets):
            target_idx = valid_target_idcs[target_idx_idx]
            for radius_idx in range(len(radii)):
                img = torch.clamp(perturbed_imgs[img_idx, target_idx, radius_idx].permute(1, 2, 0), min=0.0, max=1.0)
                img_target = targets[img_idx, target_idx]
                img_probabilities = perturbed_probabilities[img_idx, target_idx, radius_idx]

                target_conf = img_probabilities[img_target]

                ax[target_idx, radius_idx + 1].axis('off')
                ax[target_idx, radius_idx + 1].imshow(img, interpolation='lanczos')

                if num_targets == 2:
                    title = f'{class_labels[img_targets[0]]}: {img_probabilities[img_targets[0]]:.2f}\n' \
                            f'{class_labels[img_targets[1]]}: {img_probabilities[img_targets[1]]:.2f}'
                else:
                    title = f'{class_labels[img_target]}: {target_conf:.2f}'

                if show_distances:
                    pass

                ax[target_idx, radius_idx + 1].set_title(title)

                if plot_single_images:
                    _plot_single_img(img, os.path.join(single_img_dir, f'target_{target_idx}_radius_{radius_idx}.png'))
                    _plot_diff_image(img_original, img,
                                     os.path.join(single_img_dir, f'target_{target_idx}_radius_{radius_idx}_diff.png') )

        plt.tight_layout()
        if filenames is not None:
            fig.savefig(os.path.join(dir, f'{filenames[img_idx]}.png'))
            fig.savefig(os.path.join(dir, f'{filenames[img_idx]}.pdf'))
        else:
            fig.savefig(os.path.join(dir, f'{img_idx}.png'))
            fig.savefig(os.path.join(dir, f'{img_idx}.pdf'))

        plt.close(fig)

def _inner_generation(original_imgs, perturbation_targets, all_model_original_probabilities, model_descriptions, radii,
                      bs, class_labels, device, eval_dir, dataset, norm, steps, stepsize, attack_type, filenames=None,
                      plot_single_images=False, show_distanes=False, device_ids=None, verbose=False):
    num_classes = len(class_labels)
    img_dimensions = original_imgs.shape[1:]
    num_targets = perturbation_targets.shape[2]
    num_radii = len(radii)
    num_imgs = len(original_imgs)

    with torch.no_grad():

        for model_idx in range(len(model_descriptions)):
            if np.isscalar(bs):
                model_bs = bs
            else:
                model_bs = bs[model_idx]

            type, model_folder, model_checkpoint, temperature, temp = model_descriptions[model_idx]
            if verbose:
                print(f'{model_folder} {model_checkpoint} - bs {model_bs}')

            dir = f'{eval_dir}/{model_folder}_{model_checkpoint}/VisualCounterfactuals/'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            model = load_model(type, model_folder, model_checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()

            out_imgs = torch.zeros((num_imgs, num_targets, num_radii) + img_dimensions)
            out_probabilities = torch.zeros((num_imgs, num_targets, num_radii,num_classes))

            n_batches = int(np.ceil(num_imgs / model_bs))

            for batch_idx in trange(n_batches, desc=f'{model_folder} {model_checkpoint} - Batches progress'):
                sleep(0.1)
                batch_start_idx = batch_idx * model_bs
                batch_end_idx = min(num_imgs, (batch_idx + 1) * model_bs)

                batch_data = original_imgs[batch_start_idx:batch_end_idx, :]
                batch_targets = perturbation_targets[batch_start_idx:batch_end_idx, model_idx]

                for radius_idx in range(len(radii)):
                    eps = radii[radius_idx]

                    if temperature is not None and temperature < 0.2:
                        loss = 'ce-targeted-cfts-conf' if norm.lower() in ['l1', 'l2'] else 'conf' #'conf' #'logitsdiff'#'logit_max_target'
                        raise NotImplementedError()
                    else:
                        loss = 'ce-targeted-cfts-conf' if norm.lower() in ['l1', 'l2'] else 'conf' #'conf' #'logitsdiff'#'conf'
                    if verbose:
                        print('using loss', loss)

                    batch_data = batch_data.to(device)
                    batch_targets = batch_targets.to(device)

                    attack_config = create_attack_config(eps, steps, stepsize, norm,
                                                         pgd=attack_type, normalize_gradient=True)
                    att = get_adversarial_attack(attack_config, model, loss, num_classes)

                    for target_idx in range(num_targets):
                        batch_targets_i = batch_targets[:, target_idx]
                        if verbose:
                            print('filename is', filenames[target_idx])
                        #use -1 as invalid index
                        valid_batch_targets = batch_targets_i != -1
                        num_valid_batch_targets = torch.sum(valid_batch_targets).item()
                        batch_adv_samples_i = torch.zeros_like(batch_data)
                        if num_valid_batch_targets > 0:
                            if norm.lower() in ['l1', 'l2']: 
                                batch_valid_adv_samples_i = att.perturb(batch_data[valid_batch_targets],
                                                                batch_targets_i[valid_batch_targets], best_loss=True)[0].detach()
                                                                    #targeted=True).detach()
                            else:
                                att.model.T = torch.tile(torch.tensor([att.model.T], device=device),
                                                         (batch_data.shape[0], 1))
                                batch_valid_adv_samples_i = att.perturb(batch_data[valid_batch_targets],
                                                                batch_targets_i[valid_batch_targets], targeted=True).detach()
                                att.model.T = 0.7155761122703552
                            batch_adv_samples_i[valid_batch_targets] = batch_valid_adv_samples_i
                            #model.module.T = 0.7155761122703552
                            #att.model.module.T = 0.7155761122703552
                            batch_model_out_i = model(batch_adv_samples_i)
                            batch_probs_i = torch.softmax(batch_model_out_i, dim=1)
                            if verbose:
                                print('filename is', filenames[target_idx], eps)
                                print('batch targets are', batch_targets_i[valid_batch_targets]) #batch_targets_i[valid_batch_targets])
                                print('final probs are', batch_probs_i.gather(1, batch_targets_i[valid_batch_targets].reshape(-1, 1)))
                            out_imgs[batch_start_idx:batch_end_idx, target_idx, radius_idx, :] = batch_adv_samples_i.cpu().detach()
                            out_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx, :] = batch_probs_i.cpu().detach()

            model_original_probabilities = all_model_original_probabilities[:, model_idx]

            if verbose:
                print(f'min: {torch.min(out_imgs).item()} - max {torch.max(out_imgs).item()}')

            _plot_counterfactuals(dir, model_folder, model_checkpoint, original_imgs, model_original_probabilities,
                                  perturbation_targets[:, model_idx, :], out_imgs, out_probabilities, radii,
                                  class_labels, filenames=filenames, plot_single_images=plot_single_images,
                                  show_distances=show_distanes)

            #out_dict = {'model_original_probabilities': model_original_probabilities,
            #            'perturbation_targets': perturbation_targets[:, model_idx, :],
            #            'out_probabilities': out_probabilities,
            #            'radii': radii,
            #            'class_labels': class_labels
            #            }

            #out_file = os.path.join(dir, 'info.pt')
            #torch.save(out_dict, out_file)

#looks through dataloader to find examples that are misclassified  by all models,
#then for each create a counterfactual in the correct and wrong class
def visual_counterfactuals(model_descriptions, radii, dataloader, bs, num_examples, class_labels, device, eval_dir,
                           dataset, norm='L2', steps=500, stepsize=0.1, attack_type='apgd',
                           device_ids=None):
    original_imgs, perturbation_targets, all_models_original_probabilities = \
        _find_wrong_examples(model_descriptions, dataloader, num_examples, class_labels, device, dataset,
                             device_ids=device_ids)

    _inner_generation(original_imgs, perturbation_targets, all_models_original_probabilities, model_descriptions, radii,
                      bs, class_labels, device, eval_dir, dataset, norm, steps, stepsize, attack_type,
                      device_ids=device_ids)

#create counterfactuals for all models on all datapoints [N, 3, IMG_W, IMG_H] in the given classes
#target list should be a nested list where
def targeted_translations(model_descriptions, radii, imgs, target_list, bs, class_labels, device, eval_dir, dataset,
                          norm='L2', steps=500, stepsize=0.1, attack_type='apgd', show_distanes=False, filenames=None,
                          device_ids=None):
    perturbation_targets, all_models_original_probabilities = \
        _prepare_targeted_translations(model_descriptions, imgs, target_list,
                                   class_labels, device, dataset, bs, device_ids=device_ids)

    _inner_generation(imgs, perturbation_targets, all_models_original_probabilities, model_descriptions, radii, bs,
                      class_labels, device, eval_dir, dataset, norm, steps, stepsize, attack_type, filenames=filenames,
                      plot_single_images=True, show_distanes=show_distanes, device_ids=device_ids)
