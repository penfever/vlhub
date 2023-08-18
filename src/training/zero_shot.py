import logging
import random

import torch
from torch import einsum
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from open_clip import tokenize
from .precision import get_autocast

from .data import shift_cipher
from .objectnet_eval import objectnet_accuracy, accuracy_topk_subselected_and_collapsed
from .imagenet_zeroshot_data import *
from .metrics import *
try:
    from .inat_zeroshot_data import *
    from .cars_zeroshot_data import cars_classnames, cars_template
    from .food_zeroshot_data import food_classnames, food_template
    from .air_zeroshot_data import air_classnames, air_template
    from .insecta_zeroshot_data import get_insecta_classnames

except Exception as e:
    print(e)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def zero_shot_classifier(model, classnames, templates, args):
    logging.debug("In zero-shot-classifer, classnames are {}".format(classnames))
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            if args.zeroshot_scramble:
                res = []
                tlist = [t.split(" ") for t in texts]
                for l in tlist:
                    random.shuffle(l)
                    res.append(" ".join(l).strip())
            texts = tokenize(texts).to(args.device)  # tokenize
            logging.debug("In zero-shot-classifer, tokens are {}".format(classnames))
            if args.distributed and not args.horovod:
                if args.model in ["coca"]:
                    images = torch.rand(len(texts), 3, 224, 224).to(args.device)
                    class_embeddings = model.module(texts, images, return_embeddings=True)
                    class_embeddings = class_embeddings[0]
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                elif args.model in ["xclip"]:
                    images = torch.rand(len(texts), 3, model.module.image_size, model.module.image_size).to(args.device)
                    class_embeddings = model.module(texts, images, return_encodings=True)
                    if args.filip:
                        lat = model.module.to_text_latent(class_embeddings[0][:, 1:]).mean(dim=0)
                        class_embedding = l2norm(lat).mean(dim=0)
                    else:
                        class_embedding = l2norm(model.module.to_text_latent(class_embeddings[0][:, 0])).mean(dim=0)          
                else:
                    class_embeddings = model.module.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
            else:         
                if args.model in ["coca"]:
                    images = torch.rand(len(texts), 3, 224, 224).to(args.device)
                    class_embeddings = model(texts, images, return_embeddings=True)
                    class_embeddings = class_embeddings[0]
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                elif args.model in ["xclip"]:
                    images = torch.rand(len(texts), 3, model.image_size, model.image_size).to(args.device)
                    class_embeddings = model(texts, images, return_encodings=True)
                    if args.filip:
                        lat = model.to_text_latent(class_embeddings[0][:, 1:]).mean(dim=0)
                        class_embedding = l2norm(lat).mean(dim=0)
                    else:
                        class_embedding = l2norm(model.to_text_latent(class_embeddings[0][:, 0])).mean(dim=0)
                else:
                    class_embeddings = model.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classifier, dataloader, args, idx=None, split=None):
    autocast = get_autocast(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        args.isint = any([args.integer_labels, args.linear_probe])
        if args.extended_metrics:
            args.y_pred = []
            args.y_true = []
            args.logits = []
            args.image_paths = []
        for images, *target in tqdm(dataloader, unit_scale=args.batch_size):
            if split == "objectnet":
                args.img_paths = target[1]
                target = target[0]
            else:
                target = target[0]
            if args.caption_subset != "":
                if args.isint and split == "r":
                    ir_idx = get_ir_idx().tolist()
                    match_idx = sum(target==ir_idx.index(i) for i in idx).bool().nonzero(as_tuple=True)[0]
                elif args.isint and split == "a":
                    ia_idx = get_ia_idx().tolist()
                    #keep only the samples which are in passed-in class subset, using correct imagenet-a indices 
                    match_idx = sum(target==ia_idx.index(i) for i in idx).bool().nonzero(as_tuple=True)[0]
                elif args.isint and split == "objectnet":
                    obj_idx = get_obj_index().tolist()
                    match_idx = sum(target==obj_idx.index(i) for i in idx).bool().nonzero(as_tuple=True)[0]
                else:
                    match_idx = sum(target==i for i in idx).bool().nonzero(as_tuple=True)[0]
                #shave down target and images size so we skip irrelevant samples
                target = target[match_idx].to(args.device)
                images = images[match_idx].to(args.device)                
                if images.size(0) == 0:
                    continue
                if not args.isint:
                    idx_l = idx.tolist()
                    target = torch.tensor([idx_l.index(t) for t in target]).to(args.device)
                elif args.isint and split == "r":
                    ir_idx = get_ir_idx()
                    target = torch.tensor(ir_idx[target.cpu()]).to(args.device)
                elif args.isint and split == "a":
                    ia_idx = get_ia_idx()
                    target = torch.tensor(ia_idx[target.cpu()]).to(args.device)
                elif args.isint and split == "objectnet":
                    obj_idx = get_obj_index()
                    target = torch.tensor(obj_idx[target.cpu()]).to(args.device)
            else:
                images = images.to(args.device)
                try:
                    s = idx.shape
                    target = target.tolist()
                    target = torch.tensor(idx[target])
                except Exception as e:
                    pass
                target = target.to(args.device)
            #FIXME: handle larger batch sizes gracefully with gradient caching
            if args.gc:
                images = images[:min(args.gpumaxbatch, len(images)-1)]
                target = target[:min(args.gpumaxbatch, len(images)-1)]
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    if args.linear_probe:
                        logits = model.module(images)
                    elif args.integer_labels:
                        logits = model.module.visual(images)
                    elif args.model == "coca":
                        texts = torch.randint(100, (5, len(images))).to(args.device)
                        image_features = model.module(texts, images, return_embeddings=True)
                        image_features = F.normalize(image_features[1], dim=-1)
                        logits = model.module.temperature.exp() * image_features @ classifier   
                    elif args.model == "xclip":
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model.module(texts, images, return_encodings=True)
                        if args.filip:
                            image_features = l2norm(model.module.to_visual_latent(image_features[1][:, 1:])).mean(dim=1)
                        else:
                            image_features = l2norm(model.module.to_visual_latent(image_features[1][:, 0]))
                        logits = model.module.temperature.exp() * image_features @ classifier
                    else:
                        image_features = model.module.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)
                        logits = 100. * image_features @ classifier
                else:
                    if args.linear_probe:
                        logits = model(images)
                    elif args.integer_labels:
                        logits = model.visual(images)
                    elif args.model == "coca":
                        texts = torch.randint(100, (5, len(images))).to(args.device)
                        image_features = model(texts, images, return_embeddings=True)
                        image_features = F.normalize(image_features[1], dim=-1)
                        logits = model.temperature.exp() * image_features @ classifier         
                    elif args.model == "xclip":
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model(texts, images, return_encodings=True)
                        if args.filip:
                            image_features = l2norm(model.to_visual_latent(image_features[1][:, 1:])).mean(dim=1)
                        else:
                            image_features = l2norm(model.to_visual_latent(image_features[1][:, 0]))
                        #logging.info("size of image_features {}, size of classifier {}".format(image_features.size(), classifier.size()))
                        #FILIP: einsum('b t d, b i d -> b t i', *einsum_args)
                        logits = model.temperature.exp() * image_features @ classifier                             
                    else:
                        image_features = model.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)
                        logits = 100. * image_features @ classifier
            
            # remap imagenet21k pretrained model indices
            if args.isint and args.model.endswith("_in21k"):
                args.in1k_wnid = get_in1k_wnid_to_idx()
                args.in1k_wnid_rev = {v: k for k, v in args.in1k_wnid.items()}
                decoded_preds = imagenet.decode_predictions_imagenet21k(logits.cpu().detach().numpy(), top=21800)
                true_preds = []
                for pred in decoded_preds:
                    true_pred = [0 for _ in range(len(args.classnames))]
                    for poss in pred:
                        if args.in1k_wnid.get(poss[0], -1) != -1:
                            true_pred[args.in1k_wnid[poss[0]]]=poss[2]
                    true_preds.append(true_pred)
                acc1, acc5 = accuracy(torch.tensor(true_preds).to(args.device), target, topk=(1, min(5, len(args.classnames))))
                n += images.size(0)
                top1 += acc1
                top5 += acc5
                continue

            # measure accuracy with adjustments
            elif args.isint:
                args.prob_size = len(logits[0])
                #zero out logits which are not being evaluated (in VL this is handled by changing the size of the classification problem)
                if args.caption_subset != "":
                    if args.caption_subset == "insects":
                        #TODO: this is an inat to insecta class idx remap -- this should only happen if the model was trained on insecta
                        id_dict = get_insecta_id_dict()
                        id_dict = {v: k for k, v in id_dict.items()}
                        target = torch.tensor([id_dict[t] for t in target.cpu()]).to(args.device)
                    else:
                        icap_idx = get_icap_idx(args.caption_subset)
                        not_icap_idx = [i for i in range(args.prob_size) if i not in icap_idx]
                        logits[:, not_icap_idx] = float("-inf")
                if split == 'r':
                    ir_idx = get_ir_idx()
                    not_ir_idx = [i for i in range(args.prob_size) if i not in ir_idx]
                    logits[:, not_ir_idx] = float("-inf")
                if split == 'a':
                    ia_idx = get_ia_idx()
                    not_ia_idx = [i for i in range(args.prob_size) if i not in ia_idx]
                    logits[:, not_ia_idx] = float("-inf")
                if split == 'objectnet':
                    obj_idx = get_obj_index()
                    not_obj_idx = [i for i in range(args.prob_size) if i not in obj_idx]
                    logits[:, not_obj_idx] = float("-inf")
                    #TODO: for all multiclass classes, first class gets argmax of all multiclass options

            if split == "objectnet" and not args.isint:
                # if args.integer_labels:
                #     acc1, acc5 = objectnet_integer_accuracy(logits, target, args.img_paths, True, args.caption_subset)
                acc1 = objectnet_accuracy(logits, target, args.img_paths, True, args.caption_subset)
                    # acc_dict = accuracy_topk_subselected_and_collapsed(logits, target)
                if acc1 is None:
                    continue
                n += acc1[1]
                top1 += acc1[0]
                #print("top1", top1, "n", n)
                top5 += 0
                # print(acc_dict)
                # acc1 += acc_dict[0]["top1"]
                # acc5 += acc_dict[0]["top5"]
                # n += acc_dict[1]

                # acc5 = float(0.0) #TODO: implement
        
            else:
                if args.extended_metrics:
                    args.logits.append(logits.cpu().detach().numpy())
                    log_confusion_matrix(args, logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, min(5, len(args.classnames))))
                n += images.size(0)
                top1 += acc1
                top5 += acc5
                #print("top1", top1, "n", n)

    top1 = (top1 / n)
    top5 = (top5 / n)
    #TODO: debug integer labels for extended metrics
    if args.extended_metrics:
        write_confusion_matrix(args, logits, target, args.classnames)
    return top1, top5

def to_upper(l):
    return [c.upper() for c in l]

def to_lower(l):
    return [c.lower() for c in l]

def build_imagenet(args, model, in_type=""):
    isint = (args.integer_labels or args.linear_probe)
    usecaps = args.caption_subset != "" and not isint
    logging.debug("in build_imagenet, isint is {}, usecaps is {}".format(isint, usecaps))
    template = get_openai_imagenet_template()
    if args.no_ensembling:
        template = [template[0]]
    if isint:
        args.classnames = get_imagenet_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
        classifier = None
        return classifier
    if args.syn_filter:
        logging.info("Using imagenet_syn_classnames")
        base_classnames, r_classnames, a_classnames, cap_classnames, cap_r_classnames, cap_a_classnames = get_imagenet_synonym_classnames(seed=args.seed)
        if in_type == "r":
            c = cap_r_classnames if usecaps else r_classnames
        elif in_type == "a":
            c = cap_a_classnames if usecaps else a_classnames
        else:
            c = cap_classnames if usecaps else base_classnames
        return classnames_to_classifier(c, template, args, model)
    if args.def_class:
        logging.info("Using ImageNet default classnames")
        base_classnames, r_classnames, a_classnames, cap_classnames, cap_r_classnames, cap_a_classnames = get_all_imagenet_default_classnames()
        if in_type == "r":
            c = cap_r_classnames if usecaps else r_classnames
        elif in_type == "a":
            c = cap_a_classnames if usecaps else a_classnames
        else:
            c = cap_classnames if usecaps else base_classnames
        return classnames_to_classifier(c, template, args, model)        
    if in_type == "r":
        if args.ds_cipher:
            classnames = get_imagenet_r_cipher()
        elif args.ideo and usecaps:
            classnames = get_imagenet_common_ir_ideo_classnames()
        elif args.ideo:
            classnames = get_imagenet_r_ideo_classnames()
        elif usecaps:
            classnames = get_imagenet_common_ir_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
        else:
            classnames = get_imagenet_r_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
    elif in_type == "a":
        if args.ds_cipher:
            classnames = get_imagenet_a_cipher()
        elif args.ideo and usecaps:
            classnames = get_imagenet_common_ia_ideo_classnames()
        elif args.ideo:
            classnames = get_imagenet_a_ideo_classnames()
        elif usecaps:
            classnames = get_imagenet_common_ia_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
        else:
            classnames = get_imagenet_a_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
    elif in_type == "objectnet":
        if args.ds_cipher:
            classnames = get_obj_cipher()
        elif args.ideo:
            classnames = get_imagenet_obj_ideo_classnames()
        elif usecaps:
            classnames = get_imagenet_common_obj_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
        else:
            classnames = get_objectnet_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
    else:
        if args.ds_cipher:
            classnames = get_imagenet_cipher()
        elif args.ideo and usecaps:
            classnames = get_imagenet_cap_ideo_classnames()
        elif args.ideo:
            classnames = get_imagenet_ideo_classnames()
        elif usecaps:
            classnames = get_imagenet_cap_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
        else:
            classnames = get_imagenet_classnames(no_overlap=args.no_overlap, short_no_overlap=args.short_no_overlap)
    if args.zs_upper:
        classnames = to_upper(classnames)
    elif args.zs_lower:
        classnames = to_lower(classnames)
    elif args.shift_cipher:
        classnames = [shift_cipher(s, args.shift_cipher) for s in classnames]
    return classnames_to_classifier(classnames, template, args, model)

def classnames_to_classifier(classnames, template, args, model):
    logging.info("imagenet classnames first 15: {}".format(classnames[:15]))
    logging.info("length of imagenet clasnames: {}".format(len(classnames)))
    args.classnames = classnames
    logging.info('Building zero-shot classifier')
    classifier = zero_shot_classifier(model, classnames, template, args)
    return classifier

def zero_shot_eval(model, data, epoch, args):
    #logging.debug(data)
    
    results = {}
    classifier = None

    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'imagenet-r' not in data and 'imagenet-s' not in data and 'imagenet-a' not in data and 'inat2021' not in data and 'stanfordcars' not in data and 'flowers' not in data and 'food' not in data and 'objectnet' not in data and 'insecta' not in data:
        return results
    if args.zeroshot_frequency == 0:
        return results
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return results

    if 'inat2021' in data:
        isint = (args.integer_labels or args.linear_probe)
        usecaps = args.caption_subset and not isint
        if args.caption_subset != "":
            if args.caption_subset == "insects":
                args.classnames = inat_insects_classnames
                args.capsub_idx = inat_insects_idx if isint else [i for i in range(len(inat_insects_idx))]
            else:
                print("caption subset not implemented")
                raise(NotImplementedError)
        if isint:
            classifier = None
        else:
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, args.classnames, inat_template, args, args.capsub_idx)
            logging.info('Using classifier')
        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['inat2021'].dataloader, args)
        results['inat2021-top1'] = top1
        results['inat2021-top5'] = top5

        logging.info('Finished zero-shot inat2021. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'stanfordcars' in data:
        # if args.zs_upper:
        #     cars_classnames = to_upper(cars_classnames)
        # elif args.zs_lower:
        #     cars_classnames = to_lower(cars_classnames)
        logging.info("Starting zero-shot stanfordcars.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, cars_classnames, cars_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['stanfordcars'].dataloader, args)
        results['stanfordcars-top1'] = top1
        results['stanfordcars-top5'] = top5

        logging.info('Finished zero-shot stanfordcars. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'air' in data:
        logging.info("Starting zero-shot FGVC-aircraft.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, air_classnames, air_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['air'].dataloader, args)
        results['FGVC-aircraft-top1'] = top1
        results['FGVC-aircraft-top5'] = top5

        logging.info('Finished zero-shot FGVC-aircraft. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'food' in data:
            logging.info("Starting zero-shot food.")
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, food_classnames, food_template, args)

            logging.info('Using classifier')
            top1, top5 = run(model, classifier, data['food'].dataloader, args)
            results['food-top1'] = top1
            results['food-top5'] = top5

            logging.info('Finished zero-shot food. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'insecta' in data:

            isint = (args.integer_labels or args.linear_probe)
            # usecaps = args.caption_subset and not isint
            if isint:
                args.classnames = get_insecta_classnames()
                classifier = None
                # return classifier
            else:
                logging.info("Starting zero-shot insecta.")
                logging.info('Building zero-shot classifier')
                classifier = zero_shot_classifier(model, get_insecta_classnames(), inat_template, args)

            logging.info('Using classifier')
            top1, top5 = run(model, classifier, data['insecta'].dataloader, args)
            results['insecta-top1'] = top1
            results['insecta-top5'] = top5

            logging.info('Finished zero-shot insecta. Top1 was {}, top5 was {}'.format(top1, top5))

    logging.info('Starting zero-shot imagenet.')
    if args.caption_subset != "":
        logging.info("Using caption subset {}".format(args.caption_subset))
        get_icap_idx(args.caption_subset)
        get_common_ir_idx()
        get_common_ir_idx_zeroindexed()
        get_common_ia_idx()
        get_common_ia_idx_zeroindexed()
        get_common_obj_idx()
        get_common_obj_idx_zeroindexed()
    isint = args.linear_probe or args.integer_labels
    classifier = None
    imagenets = []
    if 'imagenet-val' in data:            
        if classifier is None:
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args, get_icap_idx(args.caption_subset) if args.caption_subset != "" else None)
        results['imagenet-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenet-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot val. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-v2' in data:
        if classifier is None:
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args, get_icap_idx(args.caption_subset) if args.caption_subset != "" else None)
        results['imagenetv2-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenetv2-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot v2. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-s' in data:
        if classifier is None:
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-s'].dataloader, args, get_icap_idx(args.caption_subset) if args.caption_subset != "" else None)
        results['imagenets-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenets-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot sketch. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-r' in data:
        classifier = build_imagenet(args, model, "r")
        if isint:
            top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args, get_common_ir_idx() if args.caption_subset != "" else get_ir_idx(), "r")
        else:
            top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args, get_common_ir_idx_zeroindexed() if args.caption_subset != "" else None, "r")
        results['imagenetr-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenetr-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot imagenet-r. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-a' in data:
        classifier = build_imagenet(args, model, "a")
        if isint:
            top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args, get_common_ia_idx() if args.caption_subset != "" else get_ia_idx(), "a")
        else:
            top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args, get_common_ia_idx_zeroindexed() if args.caption_subset != "" else None, "a")
        results['imageneta-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imageneta-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot imagenet-a. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'objectnet' in data:
        if classifier is None:
            classifier = build_imagenet(args, model, "objectnet")
        if isint:
            top1, top5 = run(model, classifier, data['objectnet'].dataloader, args, get_common_obj_idx() if args.caption_subset != "" else get_obj_index(), "objectnet")
        else:
            top1, top5 = run(model, classifier, data['objectnet'].dataloader, args, get_common_obj_idx_zeroindexed() if args.caption_subset != "" else None, "objectnet")
        results['objectnet-top1'] = top1
        results['objectnet-top5'] = top5        
    if results.get('imagenet-zeroshot-val-top1'):
        logging.info("computing effective robustness on imagenet")
        logging.info("len imagenets {}".format(len(imagenets)))
        try:
            imagenet_shifts = []
            for shift in ['imagenetr-zeroshot-val-top1', 'imageneta-zeroshot-val-top1', 'imagenets-zeroshot-val-top1', 'imagenetv2-zeroshot-val-top1']:
                if results.get(shift):
                    imagenet_shifts.append(results[shift])
            if len(imagenet_shifts) > 0:
                results['imagenet-average-robustness'] = np.average(imagenet_shifts)
                results['imagenet-effective-robustness'] = np.divide(np.average(imagenet_shifts), results['imagenet-zeroshot-val-top1'])
                logging.info("Average robustness over {} ImageNet shifts: {}".format(len(imagenet_shifts), results['imagenet-average-robustness']))
        except Exception as e:
            logging.info("error calculating effective robustness: ")
            logging.info(e)
    logging.info('Finished zero-shot evals')
    #save results to csv
    if args.save_results_to_csv != "":
        metrics_path = Path(args.save_results_to_csv)
        try:
            metrics_df = pd.read_csv(metrics_path)
            cols = list(metrics_df.columns)
        except:
            metrics_df = pd.DataFrame()
        results_row = {c : results[c] for c in list(results.keys())}
        # results_row = {c : results[c] for c in cols if c in list(results.keys())}
        results_row['name'] = args.model
        if len(metrics_df) > 0:
            metrics_df = metrics_df.append(pd.DataFrame(results_row,index=[len(metrics_df)+1])).fillna(0)
        else:
            metrics_df = metrics_df.append(pd.DataFrame(results_row,index=[1])).fillna(0)
        metrics_df.to_csv(metrics_path, index=False)
    return results