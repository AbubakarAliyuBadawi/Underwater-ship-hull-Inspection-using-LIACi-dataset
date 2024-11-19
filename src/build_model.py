# # build_model.py
# # location (/cluster/home/abubakb/ContMAV/src/build_model.py)

import torch
from torch import nn
from src.models.model_one_modality import OWSNetwork
from src.models.resnet import ResNet
from torchsummary import summary

def build_model(args, n_classes):
    if not args.pretrained_on_imagenet or args.last_ckpt:
        pretrained_on_imagenet = False
    else:
        pretrained_on_imagenet = True

    # set the number of channels in the encoder and for the
    # fused encoder features
    if "decreasing" in args.decoder_channels_mode:
        channels_decoder = [512, 256, 128]
        print(
            "Notice: argument --channels_decoder is ignored when "
            "--decoder_chanels_mode decreasing is set."
        )
    else:
        channels_decoder = [args.channels_decoder] * 3

    if isinstance(args.nr_decoder_blocks, int):
        nr_decoder_blocks = [args.nr_decoder_blocks] * 3
    elif len(args.nr_decoder_blocks) == 1:
        nr_decoder_blocks = args.nr_decoder_blocks * 3
    else:
        nr_decoder_blocks = args.nr_decoder_blocks
        assert len(nr_decoder_blocks) == 3

    input_channels = 3
    model = OWSNetwork(
        height=args.height,
        width=args.width,
        pretrained_on_imagenet=pretrained_on_imagenet,
        encoder=args.encoder,
        encoder_block=args.encoder_block,
        activation=args.activation,
        input_channels=input_channels,
        encoder_decoder_fusion=args.encoder_decoder_fusion,
        context_module=args.context_module,
        num_classes=n_classes,  # This will now be 11 from your args
        pretrained_dir=args.pretrained_dir,
        nr_decoder_blocks=nr_decoder_blocks,
        channels_decoder=channels_decoder,
        upsampling=args.upsampling,
    )

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("\n\n")

    model.to(device)
    print("Number of parameters:", summary(model, verbose=False).total_params)
    print("\n\n")

    # He initialization
    if args.he_init:
        module_list = []
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                continue
            for m in c.modules():
                module_list.append(m)

        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                # Safe way to check next module
                next_module = module_list[i + 1] if i + 1 < len(module_list) else None
                if (m.out_channels == n_classes or 
                    (next_module is not None and isinstance(next_module, nn.Sigmoid)) or
                    (hasattr(m, 'groups') and m.groups == m.in_channels)):
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Applied He init.")

    # Load pretrained weights if specified
    if args.finetune is not None:
        checkpoint = torch.load(args.finetune)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights for finetuning: {args.finetune}")

        print("Freeze the encoder(s).")
        for name, param in model.named_parameters():
            if "encoder_rgb" in name:
                param.requires_grad = False

    return model, device
