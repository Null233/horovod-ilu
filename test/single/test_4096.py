from torchvision import models
import numpy as np

models_used = [models.resnet18(), models.resnet50(), models.resnet101(), models.resnet152(), \
                models.vgg16(), models.vgg19(), \
                models.alexnet(), models.densenet201(), models.SqueezeNet(), \
                models.googlenet(), models.inception_v3()]
models_used_names = ['models.resnet18()', 'models.resnet50()', 'models.resnet101()', 'models.resnet152()', \
                'models.vgg16()', 'models.vgg19()', \
                'models.alexnet()', 'models.densenet201()', 'models.SqueezeNet()', \
                'models.googlenet()', 'models.inception_v3()']

for model, name in zip(models_used, models_used_names):
    m = {name:ele.size().numel() for name,ele in model.named_parameters()}
    mv = np.array([value for value in m.values()])

    mv_big = mv[np.where(mv>4096)]
    percentage = np.sum(mv_big)/np.sum(mv)
    print(f'4096:\t{name}:\t{percentage}')

    """mv_big = mv[np.where(mv>8192)]
    percentage = np.sum(mv_big)/np.sum(mv)
    print(f'8192:\t{name}:\t{percentage}')

    mv_big = mv[np.where(mv>16384)]
    percentage = np.sum(mv_big)/np.sum(mv)
    print(f'16384:\t{name}:\t{percentage}')"""
