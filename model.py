import torch
from torch import nn
from modules.utils.switchers import switcher_network

import os
from modules.utils.general import Modalities
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

from modules.networks.perceiver_pytorch import multi_modality_perceiver, Perceiver
from modules.networks.perceiver_pytorch.modalities import InputModality


def generate_model(opt):
    if opt.modalities_to_load == Modalities.OCT:

        assert opt.model_3D in [
            'MedNet10',
            'MedNet18',
            'MedNet34',
            'MedNet50',
            'MedNet101',
            'MedNet152',
            'MedNet200',
            'AHNet3D',
            'AHNet2D',
            'UNet3D',
            'VNet',
            'DenseNet3DSinglePath',
            'SkipDenseNet3D',
            'HighResNet3D',
            'DenseNet3D',
            'ResNet3dVAE',
            'HyperDenseNet',
            'DenseVoxelNet',
            'SinglePathDenseNet',
            'EfficientNet3D-B0',
            'EfficientNet3D-B1',
            'EfficientNet3D-B2',
            'EfficientNet3D-B3',
            'EfficientNet3D-B4',
            'EfficientNet3D-B5',
            'EfficientNet3D-B6',
            'EfficientNet3D-B7',
            'Perceiver']

        if 'MedNet' in opt.model_3D:
            model = switcher_network[opt.model_3D](sample_input_W=opt.input_struct_W,
                                                   sample_input_H=opt.input_struct_H,
                                                   sample_input_D=opt.input_struct_D,
                                                   sample_input_C=opt.input_struct_C,
                                                   shortcut_type=opt.model_resnet_shortcut,
                                                   no_cuda=opt.no_cuda,
                                                   num_seg_classes=opt.n_classes)
        elif 'AHNet' in opt.model_3D or 'VNet' in opt.model_3D or 'UNet3D' in opt.model_3D:
            model = switcher_network[opt.model_3D](in_channels=opt.input_C)
        elif 'DenseNet3DSinglePath' in opt.model_3D or 'ResNet3dVAE' in opt.model_3D or 'HighResNet3D' in opt.model_3D or \
                'HyperDenseNet' in opt.model_3D or 'DenseVoxelNet' in opt.model_3D or 'SinglePathDenseNet' in opt.model_3D:
            model = switcher_network[opt.model_3D](in_channels=opt.input_C, classes=opt.n_classes)
        elif 'SkipDenseNet3D' in opt.model_3D:
            model = switcher_network[opt.model_3D](in_channels=opt.input_C, classes=opt.n_classes,
                                                   num_init_features=opt.input_D)
        elif 'EfficientNet3D' in opt.model_3D:
            model_name = 'efficientnet-' + opt.model_3D.split('-')[-1].lower()
            model = switcher_network[opt.model_3D].from_name(model_name, override_params={'num_classes': opt.n_classes},
                                                             in_channels=opt.input_C)
        elif 'Perceiver' in opt.model_3D:
            model = Perceiver(
                input_channels=1,  # number of channels for each token of the input
                input_axis=3,  # number of axis for input data (2 for images, 3 for video)
                num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
                max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
                freq_base=2,
                depth=6,  # depth of net. The shape of the final attention mechanism will be:
                #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents=256, # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim=512,  # latent dimension
                cross_heads=1,  # number of heads for cross attention. paper said 1
                latent_heads=8,  # number of heads for latent self attention, 8
                cross_dim_head=64,  # number of dimensions per cross attention head
                latent_dim_head=64,  # number of dimensions per latent self attention head
                num_classes=opt.n_classes,  # output number of classes
                attn_dropout=0.,
                ff_dropout=0.,
                weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            )
        # Todo: use multiple GPUs
        # if not opt.no_cuda:
        #     if len(opt.gpu_id) > 1:
        #         model = model.cuda()
        #         model = nn.DataParallel(model, device_ids=opt.gpu_id)
        #         net_dict = model.state_dict()
        #     else:
        #         import os
        #         os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
        #         model = model.cuda()
        #         model = nn.DataParallel(model, device_ids=None)
        #         net_dict = model.state_dict()
        # else:
        net_dict = model.state_dict()

        # load pretrain
        if opt.phase != 'test' and opt.pretrain_path and os.path.isfile(opt.pretrain_path):
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)

            new_parameters = []
            for pname, p in model.named_parameters():
                for layer_name in opt.new_layer_names:
                    if pname.find(layer_name) >= 0:
                        new_parameters.append(p)
                        break

            new_parameters_id = list(map(id, new_parameters))
            base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
            parameters = {'base_parameters': base_parameters,
                          'new_parameters': new_parameters}

            return model, parameters

        return model, model.parameters()

    elif opt.modalities_to_load == Modalities.Fundus:

        assert opt.model_2D in [
            'adv_inception_v3',
            'cait_m36_384',
            'cait_m48_448',
            'cait_s24_224',
            'cait_s24_384',
            'cait_s36_384',
            'cait_xs24_384',
            'cait_xxs24_224',
            'cait_xxs24_384',
            'cait_xxs36_224',
            'cait_xxs36_384',
            'coat_lite_mini',
            'coat_lite_tiny',
            'cspdarknet53',
            'cspresnet50',
            'cspresnext50',
            'densenet121',
            'densenet161',
            'densenet169',
            'densenet201',
            'densenetblur121d',
            'dla34',
            'dla46_c',
            'dla46x_c',
            'dla60',
            'dla60_res2net',
            'dla60_res2next',
            'dla60x',
            'dla60x_c',
            'dla102',
            'dla102x',
            'dla102x2',
            'dla169',
            'dm_nfnet_f0',
            'dm_nfnet_f1',
            'dm_nfnet_f2',
            'dm_nfnet_f3',
            'dm_nfnet_f4',
            'dm_nfnet_f5',
            'dm_nfnet_f6',
            'dpn68',
            'dpn68b',
            'dpn92',
            'dpn98',
            'dpn107',
            'dpn131',
            'eca_nfnet_l0',
            'eca_nfnet_l1',
            'ecaresnet26t',
            'ecaresnet50d',
            'ecaresnet50d_pruned',
            'ecaresnet50t',
            'ecaresnet101d',
            'ecaresnet101d_pruned',
            'ecaresnet269d',
            'ecaresnetlight',
            'efficientnet_b0',
            'efficientnet_b1',
            'efficientnet_b1_pruned',
            'efficientnet_b2',
            'efficientnet_b2_pruned',
            'efficientnet_b3',
            'efficientnet_b3_pruned',
            'efficientnet_b4',
            'efficientnet_el',
            'efficientnet_el_pruned',
            'efficientnet_em',
            'efficientnet_es',
            'efficientnet_es_pruned',
            'efficientnet_lite0',
            'efficientnetv2_rw_s',
            'ens_adv_inception_resnet_v2',
            'ese_vovnet19b_dw',
            'ese_vovnet39b',
            'fbnetc_100',
            'gernet_l',
            'gernet_m',
            'gernet_s',
            'ghostnet_100',
            'gluon_inception_v3',
            'gluon_resnet18_v1b',
            'gluon_resnet34_v1b',
            'gluon_resnet50_v1b',
            'gluon_resnet50_v1c',
            'gluon_resnet50_v1d',
            'gluon_resnet50_v1s',
            'gluon_resnet101_v1b',
            'gluon_resnet101_v1c',
            'gluon_resnet101_v1d',
            'gluon_resnet101_v1s',
            'gluon_resnet152_v1b',
            'gluon_resnet152_v1c',
            'gluon_resnet152_v1d',
            'gluon_resnet152_v1s',
            'gluon_resnext50_32x4d',
            'gluon_resnext101_32x4d',
            'gluon_resnext101_64x4d',
            'gluon_senet154',
            'gluon_seresnext50_32x4d',
            'gluon_seresnext101_32x4d',
            'gluon_seresnext101_64x4d',
            'gluon_xception65',
            'hardcorenas_a',
            'hardcorenas_b',
            'hardcorenas_c',
            'hardcorenas_d',
            'hardcorenas_e',
            'hardcorenas_f',
            'hrnet_w18',
            'hrnet_w18_small',
            'hrnet_w18_small_v2',
            'hrnet_w30',
            'hrnet_w32',
            'hrnet_w40',
            'hrnet_w44',
            'hrnet_w48',
            'hrnet_w64',
            'ig_resnext101_32x8d',
            'ig_resnext101_32x16d',
            'ig_resnext101_32x32d',
            'ig_resnext101_32x48d',
            'inception_resnet_v2',
            'inception_v3',
            'inception_v4',
            'legacy_senet154',
            'legacy_seresnet18',
            'legacy_seresnet34',
            'legacy_seresnet50',
            'legacy_seresnet101',
            'legacy_seresnet152',
            'legacy_seresnext26_32x4d',
            'legacy_seresnext50_32x4d',
            'legacy_seresnext101_32x4d',
            'mixer_b16_224',
            'mixer_b16_224_in21k',
            'mixer_l16_224',
            'mixer_l16_224_in21k',
            'mixnet_l',
            'mixnet_m',
            'mixnet_s',
            'mixnet_xl',
            'mnasnet_100',
            'mobilenetv2_100',
            'mobilenetv2_110d',
            'mobilenetv2_120d',
            'mobilenetv2_140',
            'mobilenetv3_large_100',
            'mobilenetv3_large_100_miil',
            'mobilenetv3_large_100_miil_in21k',
            'mobilenetv3_rw',
            'nasnetalarge',
            'nf_regnet_b1',
            'nf_resnet50',
            'nfnet_l0',
            'pit_b_224',
            'pit_b_distilled_224',
            'pit_s_224',
            'pit_s_distilled_224',
            'pit_ti_224',
            'pit_ti_distilled_224',
            'pit_xs_224',
            'pit_xs_distilled_224',
            'pnasnet5large',
            'regnetx_002',
            'regnetx_004',
            'regnetx_006',
            'regnetx_008',
            'regnetx_016',
            'regnetx_032',
            'regnetx_040',
            'regnetx_064',
            'regnetx_080',
            'regnetx_120',
            'regnetx_160',
            'regnetx_320',
            'regnety_002',
            'regnety_004',
            'regnety_006',
            'regnety_008',
            'regnety_016',
            'regnety_032',
            'regnety_040',
            'regnety_064',
            'regnety_080',
            'regnety_120',
            'regnety_160',
            'regnety_320',
            'repvgg_a2',
            'repvgg_b0',
            'repvgg_b1',
            'repvgg_b1g4',
            'repvgg_b2',
            'repvgg_b2g4',
            'repvgg_b3',
            'repvgg_b3g4',
            'res2net50_14w_8s',
            'res2net50_26w_4s',
            'res2net50_26w_6s',
            'res2net50_26w_8s',
            'res2net50_48w_2s',
            'res2net101_26w_4s',
            'res2next50',
            'resnest14d',
            'resnest26d',
            'resnest50d',
            'resnest50d_1s4x24d',
            'resnest50d_4s2x40d',
            'resnest101e',
            'resnest200e',
            'resnest269e',
            'resnet18',
            'resnet18d',
            'resnet26',
            'resnet26d',
            'resnet34',
            'resnet34d',
            'resnet50',
            'resnet50d',
            'resnet101d',
            'resnet152d',
            'resnet200d',
            'resnetblur50',
            'resnetrs50',
            'resnetrs101',
            'resnetrs152',
            'resnetrs200',
            'resnetrs270',
            'resnetrs350',
            'resnetrs420',
            'resnetv2_50x1_bitm',
            'resnetv2_50x1_bitm_in21k',
            'resnetv2_50x3_bitm',
            'resnetv2_50x3_bitm_in21k',
            'resnetv2_101x1_bitm',
            'resnetv2_101x1_bitm_in21k',
            'resnetv2_101x3_bitm',
            'resnetv2_101x3_bitm_in21k',
            'resnetv2_152x2_bitm',
            'resnetv2_152x2_bitm_in21k',
            'resnetv2_152x4_bitm',
            'resnetv2_152x4_bitm_in21k',
            'resnext50_32x4d',
            'resnext50d_32x4d',
            'resnext101_32x8d',
            'rexnet_100',
            'rexnet_130',
            'rexnet_150',
            'rexnet_200',
            'selecsls42b',
            'selecsls60',
            'selecsls60b',
            'semnasnet_100',
            'seresnet50',
            'seresnet152d',
            'seresnext26d_32x4d',
            'seresnext26t_32x4d',
            'seresnext50_32x4d',
            'skresnet18',
            'skresnet34',
            'skresnext50_32x4d',
            'spnasnet_100',
            'ssl_resnet18',
            'ssl_resnet50',
            'ssl_resnext50_32x4d',
            'ssl_resnext101_32x4d',
            'ssl_resnext101_32x8d',
            'ssl_resnext101_32x16d',
            'swin_base_patch4_window7_224',
            'swin_base_patch4_window7_224_in22k',
            'swin_base_patch4_window12_384',
            'swin_base_patch4_window12_384_in22k',
            'swin_large_patch4_window7_224',
            'swin_large_patch4_window7_224_in22k',
            'swin_large_patch4_window12_384',
            'swin_large_patch4_window12_384_in22k',
            'swin_small_patch4_window7_224',
            'swin_tiny_patch4_window7_224',
            'swsl_resnet18',
            'swsl_resnet50',
            'swsl_resnext50_32x4d',
            'swsl_resnext101_32x4d',
            'swsl_resnext101_32x8d',
            'swsl_resnext101_32x16d',
            'tf_efficientnet_b0',
            'tf_efficientnet_b0_ap',
            'tf_efficientnet_b0_ns',
            'tf_efficientnet_b1',
            'tf_efficientnet_b1_ap',
            'tf_efficientnet_b1_ns',
            'tf_efficientnet_b2',
            'tf_efficientnet_b2_ap',
            'tf_efficientnet_b2_ns',
            'tf_efficientnet_b3',
            'tf_efficientnet_b3_ap',
            'tf_efficientnet_b3_ns',
            'tf_efficientnet_b4',
            'tf_efficientnet_b4_ap',
            'tf_efficientnet_b4_ns',
            'tf_efficientnet_b5',
            'tf_efficientnet_b5_ap',
            'tf_efficientnet_b5_ns',
            'tf_efficientnet_b6',
            'tf_efficientnet_b6_ap',
            'tf_efficientnet_b6_ns',
            'tf_efficientnet_b7',
            'tf_efficientnet_b7_ap',
            'tf_efficientnet_b7_ns',
            'tf_efficientnet_b8',
            'tf_efficientnet_b8_ap',
            'tf_efficientnet_cc_b0_4e',
            'tf_efficientnet_cc_b0_8e',
            'tf_efficientnet_cc_b1_8e',
            'tf_efficientnet_el',
            'tf_efficientnet_em',
            'tf_efficientnet_es',
            'tf_efficientnet_l2_ns',
            'tf_efficientnet_l2_ns_475',
            'tf_efficientnet_lite0',
            'tf_efficientnet_lite1',
            'tf_efficientnet_lite2',
            'tf_efficientnet_lite3',
            'tf_efficientnet_lite4',
            'tf_efficientnetv2_b0',
            'tf_efficientnetv2_b1',
            'tf_efficientnetv2_b2',
            'tf_efficientnetv2_b3',
            'tf_efficientnetv2_l',
            'tf_efficientnetv2_l_in21ft1k',
            'tf_efficientnetv2_l_in21k',
            'tf_efficientnetv2_m',
            'tf_efficientnetv2_m_in21ft1k',
            'tf_efficientnetv2_m_in21k',
            'tf_efficientnetv2_s',
            'tf_efficientnetv2_s_in21ft1k',
            'tf_efficientnetv2_s_in21k',
            'tf_inception_v3',
            'tf_mixnet_l',
            'tf_mixnet_m',
            'tf_mixnet_s',
            'tf_mobilenetv3_large_075',
            'tf_mobilenetv3_large_100',
            'tf_mobilenetv3_large_minimal_100',
            'tf_mobilenetv3_small_075',
            'tf_mobilenetv3_small_100',
            'tf_mobilenetv3_small_minimal_100',
            'tnt_s_patch16_224',
            'tresnet_l',
            'tresnet_l_448',
            'tresnet_m',
            'tresnet_m_448',
            'tresnet_m_miil_in21k',
            'tresnet_xl',
            'tresnet_xl_448',
            'tv_densenet121',
            'tv_resnet34',
            'tv_resnet50',
            'tv_resnet101',
            'tv_resnet152',
            'tv_resnext50_32x4d',
            'vgg11',
            'vgg11_bn',
            'vgg13',
            'vgg13_bn',
            'vgg16',
            'vgg16_bn',
            'vgg19',
            'vgg19_bn',
            'vit_base_patch16_224',
            'vit_base_patch16_224_in21k',
            'vit_base_patch16_224_miil',
            'vit_base_patch16_224_miil_in21k',
            'vit_base_patch16_384',
            'vit_base_patch32_224_in21k',
            'vit_base_patch32_384',
            'vit_base_r50_s16_224_in21k',
            'vit_base_r50_s16_384',
            'vit_deit_base_distilled_patch16_224',
            'vit_deit_base_distilled_patch16_384',
            'vit_deit_base_patch16_224',
            'vit_deit_base_patch16_384',
            'vit_deit_small_distilled_patch16_224',
            'vit_deit_small_patch16_224',
            'vit_deit_tiny_distilled_patch16_224',
            'vit_deit_tiny_patch16_224',
            'vit_large_patch16_224',
            'vit_large_patch16_224_in21k',
            'vit_large_patch16_384',
            'vit_large_patch32_224_in21k',
            'vit_large_patch32_384',
            'vit_small_patch16_224',
            'wide_resnet50_2',
            'wide_resnet101_2',
            'xception',
            'xception41',
            'xception65',
            'xception71',
            'Perceiver'
        ]
        if 'Perceiver' in opt.model_2D:
            model = Perceiver(
                input_channels=3,  # number of channels for each token of the input
                input_axis=2,  # number of axis for input data (2 for images, 3 for video)
                num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
                max_freq=10.,  # maximum frequency, hyperparameter depending on how fine the data is
                freq_base=2,
                depth=6,  # depth of net. The shape of the final attention mechanism will be:
                #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents=256,
                # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim=512,  # latent dimension
                cross_heads=1,  # number of heads for cross attention. paper said 1
                latent_heads=8,  # number of heads for latent self attention, 8
                cross_dim_head=64,  # number of dimensions per cross attention head
                latent_dim_head=64,  # number of dimensions per latent self attention head
                num_classes=opt.n_classes,  # output number of classes
                attn_dropout=0.,
                ff_dropout=0.,
                weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            )
        else:
            model = create_model(
                opt.model_2D,
                pretrained=opt.pretrained,
                num_classes=opt.n_classes,
                drop_rate=opt.drop,
                drop_connect_rate=opt.drop_connect,  # DEPRECATED, use drop_path
                drop_path_rate=opt.drop_path,
                drop_block_rate=opt.drop_block,
                global_pool=opt.gp,
                bn_tf=opt.bn_tf,
                bn_momentum=opt.bn_momentum,
                bn_eps=opt.bn_eps,
                scriptable=opt.torchscript,
                checkpoint_path=opt.initial_checkpoint)

        return model, model.parameters()

    elif opt.modalities_to_load == Modalities.OCT_FUNDUS:
        oct_modality = InputModality(
            name='video',
            input_channels=1,  # number of channels for each token of the input
            input_axis=3,  # number of axes, 3 for video)
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
        )
        image_modality = InputModality(
            name='image',
            input_channels=3,  # number of channels for each token of the input
            input_axis=2,  # number of axes, 2 for images
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
        )
        model = multi_modality_perceiver.MultiModalityPerceiver(
        # model = multi_modality_perceiver.MultiModalityPerceiverNoPooling(
            modalities=(oct_modality, image_modality),
            depth=2,  # TODO modify it appropriately # depth of net
            num_latents=12,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=64,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=opt.n_classes,  # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=True
            # whether to weight tie layers (optional, as indicated in the diagram)
        )
        return model, model.parameters()
