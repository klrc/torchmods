from torch.utils.data import Dataset, DataLoader


def is_enum(obj):
    return type(obj) is tuple or type(obj) is list


def datasets(dataset_class, transforms=(None, None, None), **kwargs):
    datasets_collection = [
        dataset_class(train=True, transform=transforms[0], **kwargs),
        dataset_class(train=False, transform=transforms[1], **kwargs),
        dataset_class(train=False, transform=transforms[2], **kwargs),
    ]
    # check transforms
    assert len(transforms) == 3
    for d in datasets_collection:
        try:
            _ = d.__getitem__(0)
        except Exception as e:
            raise e
    return datasets_collection


def dataloaders(datasets,  **kwargs):
    # check datasets input
    assert len(datasets) == 3
    for x in datasets:
        assert issubclass(type(x), Dataset)
        try:
            x.__getitem__(0)
        except Exception as e:
            raise e
    trainset, testset, valset = datasets
    trainloader = DataLoader(trainset, shuffle=True, **kwargs)
    testloader = DataLoader(testset, **kwargs)
    valloader = DataLoader(valset, **kwargs)
    return trainloader, testloader, valloader
