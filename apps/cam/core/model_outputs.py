#
from apps.cam.core.feature_extractor import FeatureExtractor

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers, headers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            print('name: {0};'.format(name))
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            '''
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif 'brand_clfr' in name.lower():
                x = module(x)
                break
            '''
            else:
                x = module(x)
        x = header['avgpool'](x)
        x = x.view(x.size(0),-1)
        x = header['classifier'](x)
        
        return target_activations, x