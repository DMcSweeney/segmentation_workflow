"""
Useful functions for onnx conversion
"""
import torchvision.models.segmentation as tvms
import segmentation_models_pytorch as smpt
import torch
import onnxruntime
import numpy as np



#~ ======== Get models
def get_torchvision_model(name, classes):
    """
    Figure out which case we have and load the model & weights
    """

    return tvms.__dict__[name](num_classes=classes)

def get_segmentation_models_model(name, classes, in_channels):
    print(smpt.__dict__)
    return smpt.__dict__[name](classes=classes, in_channels=in_channels)

def get_custom_model(name):
    print('Collecting ', name)
    model_bank = {}
    return model_bank[name] if name in model_bank else None

def check_predictions(onnx_modelpath, pytorch_model, dummy_input):
    #~ Check prediction 
    torch_out = pytorch_model(dummy_input)

    ort_session = onnxruntime.InferenceSession(onnx_modelpath)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(np.round(to_numpy(sigmoid(torch_out['out']))), 
    np.round(to_numpy(sigmoid(torch.tensor(ort_outs[0])))), rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")