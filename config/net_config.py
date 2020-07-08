if args.arch == "frankenstein":
    model = frankestein(num_classes=2, in_channels=3)
    args.model = model.to(device)
    sequence_first = False
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.00001)
    base_size = None
    xCrop = 224
    yCrop = 224
    crop_time = 64
    BATCH_SIZE = 4

if args.arch == "ResNet2+":
    model = R2Plus1DClassifier(num_classes=2)
    model = model.to(device)
    sequence_first = False
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    base_size = None
    xCrop = 224
    yCrop = 224
    crop_time = 32
    BATCH_SIZE = 1

if args.arch == "I3D":
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.load_state_dict(torch.load('models/i3d_rgb_imagenet.pt'))
    model.replace_logits(num_classes)
    model = model.to(device)
    sequence_first = False
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    base_size = None
    xCrop = 224
    yCrop = 224
    crop_time = 64
    BATCH_SIZE = 4


if args.arch == "ResNet3D":
    BATCH_SIZE = 8
    args.test_batch_size = 16
    crop_time = 64
    model = resnet3D34(num_classes=400, shortcut_type='A',
                                    sample_size=112, sample_duration=crop_time,
                                    last_fc=True)
    model_data = torch.load("models/resnet-34-kinetics.pth")
    state_dict = model_data['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    model.fc = nn.Linear(8192, num_classes)
    model = model.to(device)
    sequence_first = False
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    base_size = 256
    xCrop = 224
    yCrop = 224


if args.arch == "P3D":
    model = P3D199(pretrained=True,num_classes=2)
    sequence_first = False
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    base_size = None
    xCrop = None
    yCrop = None
    crop_time = 16
    BATCH_SIZE = 20



if args.arch == "Attention":
    model = attentionModel(dropout=dropout, num_classes=num_classes)
    sequence_first = True
    model = model.to(device)
    optimizer = optim.SGD(itertools.chain(*[model.parameters(),model.lstm_cell.parameters()]), lr=0.001, momentum=0.9, weight_decay=0.00001)
    base_size = 256
    xCrop = yCrop = 224
    crop_time = 64
    BATCH_SIZE = 1

if args.arch == "ResNet2D+1":
    model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_8_kinetics", num_classes=400, pretrained=True):
    sequence_first = True
    model = model.to(device)
    optimizer = optim.SGD(itertools.chain(*[model.parameters()]), lr=0.001, momentum=0.9, weight_decay=0.00001)
    base_size = 256
    xCrop = yCrop = 224
    crop_time = 4
    BATCH_SIZE = 1
