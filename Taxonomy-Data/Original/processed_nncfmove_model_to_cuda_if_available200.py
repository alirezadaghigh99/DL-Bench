def move_model_to_cuda_if_available(model):
    if torch.cuda.is_available():
        model.cuda()
    return next(iter(model.parameters())).device