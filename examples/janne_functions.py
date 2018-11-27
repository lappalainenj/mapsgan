def reassign_layers(model, last_function = ['ReLU', 'Sigmoid']):
    """Takes each module in a model and restructures the functions into modules
    called layer[i] according to the position i of a function specified in 
    last_function. E.g. one would like to have layers always end with an
    activation function, then last_function should be filled with all names of
    activation functions in the model.
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        last_function (list of str): Function names that are used to identify 
            the end of a layer.
    """
    Concat(model)
    module = model._modules.pop('concatenated')
    functions, layers = _get_functions(module, last_function) # 
    layers = _sort_layers(module, functions, layers)
    _fill_modules(model, layers)
    

class Concat():
    """Concats all modules of the model.
    
    Note: Be cautious with recursive/residual networks. Certain blocks that 
        define the layer architecture by the forward function should stay intact.
    
    Todo:
        * Test Concat Class on all pretrained Pytorch Modules, if necessary extend
        a in method _get_module.
    
    Args:
        model (torch.nn.Module): The module which submodules will
            be concatenated to a single module called concatenated.
    
    Attributes:
        model (torch.nn.Module): The module which submodules will
            be concatenated to a single module called concatenated.
        list_of_layers (list): List to temporally store the submodules of the 
            network.
    """
    def __init__(self, model):
        self.model = model
        self.list_of_layers = []
        self._get_modules(model)
        self._concat()
    
    def _get_modules(self, model):
        """Recursively puts all submodules into list_of_layers.
        Note: Submodules in residual/recursive networks should not be split!"""
        from sys import modules
        from inspect import getmembers, isclass
        #enable for resnet, tricky: shallow and nested nn modules / keep BasicBlocks
        a = [x[1] for x in getmembers(modules[models.resnet.__name__], isclass)]
        for module in model.children():
            if not list(module.children()) or isinstance(module, tuple(a)):
                self.list_of_layers.append(module)
            else:
                self._get_modules(module)
        
    def _concat(self):
        modules = [module for module in self.model._modules]
        for module in modules:
            self.model._modules.pop(module)
        self.model._modules['concatenated'] = nn.Sequential(*self.list_of_layers)


def _get_functions(module, last_function):
    """Identifies the positions i of the functions, specified in last_function,
    and thus defining the end of the new layers.
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        last_function (list of str): Function names that are used to identify 
            the end of a layer.
    
    Returns:
        functions (list): A list containing the keys of all functions that were
            identified in the submodule.
        layers (dict): A dict containing a key, value pair of a layer index and
            an empty list.
    """
    functions = []
    layers = {}
    lastsubmodule = False
    for key, submodule in module._modules.items():
        if key == next(reversed(module._modules)):
            lastsubmodule = True
        match = [function in submodule.__str__() for function in last_function]
        if any(match+[lastsubmodule]):
            functions.append(key)
    layers.update({i+1:[] for i in range(len(functions))})
    return functions, layers


def _sort_layers(module, functions, layers):
    """Constructs the new layers. The modules are filled into the lists that
    describe the new layers in ascending order up to the final function. 
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        functions (list): A list containing the keys of all functions that were
            identified in the submodule.
        layers (dict): A dict containing a key, value pair of a layer index and
            an empty list.
    
    Returns:
        layers (dict): A dict containing a key, value pair of a layer index and 
            a list with all respective torch.nn.functions.
    """
    for i, key in enumerate(functions):
        for j in range(int(key)+1):
            try:
                layers[i+1].append(module._modules.pop(str(j)))                
            except:
                pass
    return layers


def _fill_modules(model, layers):
    """Creates the new nn.modules according to the extracted layers and assigns
    them to the model. 
   
    Args:
        model (torch.nn.Module): The model that needs to be restructured.
        layers (dict): A dict containing a key, value pair of a layer index and 
            a list with all respective torch.nn.functions.
    """
    for key, value in layers.items():
        model._modules['layer%s'%(key)] = nn.Sequential(*value)
        
def _forward(self, x):
    """A generic forward function, which stores the activations specified in 
    extracted_layers. It works for pure forward networks including transitions
    of convolution and fully connected layers. 
    
    Note: Tested with AlexNet, VGG. ResNet with its BasicBlocks kept intact.
    """
    outputs = {}
    for name, module in self._modules.items():
        for subname, submodule in module._modules.items():
            try:
                x = submodule(x)
            except:
                x = x.view(x.size(0), -1)
                x = submodule(x)
        if self.extract_layers and name in self.extract_layers:
            outputs[name] = x
            if name == self.extract_layers[-1]:
                return outputs
    if outputs:
        return outputs
    return x    


