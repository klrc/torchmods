# Torchmods Supports Pack for Pytorch
## Getting Started
### installation
    pip install torchmods -U -i https://pypi.org/simple

### use torchmods
    import torchmods as mods

#### mods: ssh support
torchmods have some simple ssh control module, to use it:

    # generate your key & keep it.
    key = mods.ssh.keygen('your_server_ip', 'username', 'password')
    print(key)

    # ssh time!
    with mods.ssh(key) as f:
        print(f.call('ls -ll'))                              # run command & get opstream
        f.upload('sample.txt', './Desktop/sample_copy.txt')  # sftp upload
        f.cd('./Desktop')                                    # global change-dir operation
        f.download('sample_copy.txt', 'sample_copy.txt')     # sftp download

#### mods: device control support
with torchmods, a remote server can be virtualized as a device with `device.push()`:

    # connect to remote device
    device = mods.Device(key, 'your_root')

    # push task to your device
    device.push('your/task/dir')

everything is done, you are free to leave or push anything later.

you can check the status with `device.check_something()`(NOT implemented now):

    # not implemented yet
    device.check_something()

all tasks are executed and packed in the `/output` dir, you can pull it with `device.pull()`

to specify conda environment, running args, task control or anything else, you can just customize the core script:

    device.core.server_script = './myscript.sh'

the default script are stored in torchmods/device/server.sh:

    #!/bin/sh

    ######################################
    #                                    #
    #           Magic by sh              #
    #                                    #
    ######################################

    # check if there is new tasks
    source activate dl
    while ! [[ -z `ls -A ../input` ]]
    do
        cd ../input
        for file in ./*
        do  
            # clean virtual env
            rm ../vm/*
            # decompress task & chroot
            full_file_name=${file##*/}
            file_name=${full_file_name%.tar.bz2}
            tar -jxf $file -C ../vm/
            rm $file
            # run task
            cd ../vm
            echo True >../server/awake
            nohup python -u main.py >history 2>&1 & pid=$!
            echo $pid >../server/pid
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] running task ${file_name}, pid=${pid}"
            wait $pid
            # compress task outputs
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] task ${file_name} finished, compress outputs"
            echo False >../server/awake
            tar -jcf ../output/${file_name}.tar.bz2 .
            # recover dir
            cd ../input
        done
        # recover dir
        cd ../server
    done

#### mods: datasets support
we have API supports for datasets in CV research field, which is compatible with original pytorch, sample code:

    # download not supported yet, baidu netdisk sucks :(
    train_dataset = mods.datasets.LDL(your_root, train=True)
    test_dataset = mods.datasets.LDL(your_root, train=False)

    from torch.utils.data import DataLoader
    trainloader = Dataloader(train_dataset, batch_size=16)
    testloader = Dataloader(test_dataset, batch_size=16)
    ...

or one-line-training with our full API, we offer everything even including standard loss, its just too lazy.

    from torchvision.models import resnet50
    env = EnvLDL()
    model = resnet50().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    env.train(model, optimizer)

debugging and evaluation are as simple:

    env.check(model, optimizer)
    env.eval(model)

#### mods: mods.nn support
we add a few things to `torch.nn` such as CanberLoss (fully compatible with origin pytorch):

    # import torch.nn as nn
    import mods.nn as nn

    loss = nn.SoftmaxWrapper(nn.CorrelationCoefficientsLoss()).cuda()
    x = torch.rand(4, 1, 30, 40).cuda()
    target = torch.rand(4, 1, 30, 40).cuda()
    print(loss(x, target))

#### mods: visualize support
supports for tensor-to-numpy convertion or visualization.

- denormalize(tensor)
- heatmap(tensor)
- NumpyImage(tensor)
- Gif()
- ImageGrid()
