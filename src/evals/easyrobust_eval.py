try:
    from easyrobust.benchmarks import *
    er = True
except ImportError:
    print('easyrobust not available')
    er = False

def easyrobust_eval(model, args):
    if not er:
        print('easyrobust not available')
        return
    top1_si = evaluate_stylized_imagenet(model, args.stylized_imagenet)
    top1_c, _ = evaluate_imagenet_c(model, args.imagenet_c)
    # objectnet is optional since it spends a lot of disk storage. we skip it here. 
    top1_obj = evaluate_objectnet(model, args.objectnet)
    return {
        'stylized_imagenet_top1': top1_si,
        'imagenet_c_top1': top1_c,
        'objectnet_top1': top1_obj,
    }