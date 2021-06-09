import torch
import numpy as np



def load_darknet_weights( weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    # file = Path(weights).name
    # if file == 'darknet53.conv.74':
    #     cutoff = 75
    # elif file == 'yolov3-tiny.conv.15':
    #     cutoff = 15
    weights
    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(module_defs[:cutoff], module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def read_model(model):
    # weight_file='/home/mayank_sati/codebase/python/lidar/second.pytorch/second/mayank_scripts/new_pp_model.pb'
    start_epoch = 0
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()
    # print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    # state_dict_ = checkpoint['state_dict']
    state_dict = {}
    layer = model_state_dict['voxel_feature_extractor.pfn_layers.0.conv_layer_1.weight']
    myweight = layer.cpu().detach().numpy()
    myweight2 = myweight.squeeze()
    print(myweight2)

    # print(myweight)
    # print(layer.bias)
    1


def load_model_2(model, model_path, opt, optimizer=None):
    # weight_file='/home/mayank_sati/codebase/python/lidar/second.pytorch/second/mayank_scripts/new_pp_model.pb'
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    for k in state_dict:
        for j in model_state_dict:
            if (k=="voxel_feature_extractor.pfn_layers.0.linear.weight" and j=="voxel_feature_extractor.pfn_layers.0.conv_layer_1.weight"):
                print("gotit")
                layer1 = state_dict['voxel_feature_extractor.pfn_layers.0.linear.weight']
                layer1.view(64, 9, 1, 1)
                myweight = layer1.cpu().detach().numpy()
                model_state_dict["voxel_feature_extractor.pfn_layers.0.conv_layer_1.weight"]= layer1.view(64, 9, 1, 1)
                myweight2=model_state_dict["voxel_feature_extractor.pfn_layers.0.conv_layer_1.weight"].squeeze().cpu().numpy()
                # myweight2 = myweight.squeeze()
                # print(myweight2)
    # check loaded parameters and created model parameters
    # for k in state_dict:
    #     # if k in model_state_dict:
    #         if (state_dict[k].shape != model_state_dict[k].shape):
    #             if opt.reuse_hm:
    #                 print('Reusing parameter {}, required shape{}, ' \
    #                       'loaded shape{}.'.format(
    #                     k, model_state_dict[k].shape, state_dict[k].shape))
    #                 if state_dict[k].shape[0] < state_dict[k].shape[0]:
    #                     model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
    #                 else:
    #                     model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
    #                 state_dict[k] = model_state_dict[k]
    #             else:
    #                 print('Skip loading parameter {}, required shape{}, ' \
    #                       'loaded shape{}.'.format(
    #                     k, model_state_dict[k].shape, state_dict[k].shape))
    #                 state_dict[k] = model_state_dict[k]
    #     # else:
    #     #     print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and opt.resume:
        if 'optimizer' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = opt.lr
            for step in opt.lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model



def load_model_3(model, model_path, opt, optimizer=None):
    # weight_file='/home/mayank_sati/codebase/python/lidar/second.pytorch/second/mayank_scripts/new_pp_model.pb'
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # layer = checkpoint['voxel_feature_extractor.pfn_layers.0.conv_layer_1.weight']
    # myweight = layer.cpu().detach().numpy()
    # myweight2 = myweight.squeeze()
    # print(myweight2)
    #
    # # print(myweight)
    # print(layer.bias)
    1
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if (state_dict[k].shape != model_state_dict[k].shape):
                if opt.reuse_hm:
                    print('Reusing parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    if state_dict[k].shape[0] < state_dict[k].shape[0]:
                        model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
                    else:
                        model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
                    state_dict[k] = model_state_dict[k]
                else:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and opt.resume:
        if 'optimizer' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = opt.lr
            for step in opt.lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


if __name__ == '__main__':
    # weight_file='/home/mayank_sati/codebase/python/lidar/second.pytorch/checkpoint/voxelnet-140670.tckpt'
    # weight_file='/home/mayank_sati/codebase/python/lidar/second.pytorch/second/checkpoint/new_pp_model.tckpt'
    weight_file='/home/mayank_sati/codebase/python/lidar/second.pytorch/checkpoint/mayank_old_pp_trained.pb'
    # load_darknet_weights(weight_file)
    load_model_2(1,weight_file,1)