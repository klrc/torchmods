import os
from visdom import Visdom


def visdom():
    # check visdom connection
    viz = Visdom(server='http://127.0.0.1', port=8097)
    if not viz.check_connection(timeout_seconds=5):
        os.system('nohup python -m visdom.server &')
    assert viz.check_connection(timeout_seconds=5)
    return viz


# from PIL import Image
# from torchvision import transforms

# viz = visdom_server()
# viz.image(
#     transforms.ToTensor()(Image.open('/home/sh/test.jpg').convert('RGB')),
#     opts={
#         'title': 'TestHakase',
#         'showlegend': True
#     }
# )
