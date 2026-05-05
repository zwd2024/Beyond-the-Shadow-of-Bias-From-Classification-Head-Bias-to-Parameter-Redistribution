import torch
def BiasShift(
    args,
    model,
    device,
):
    model.to(device)

    # 仅修改遗忘类别的 bias (Only modify the bias of the forgotten category)
    with torch.no_grad():
        if hasattr(model, 'fc') and model.fc.bias is not None:
            model.fc.bias[args.unlearn_class] -= args.bias_shift_value
        else:
            raise ValueError("Model does not have fc.bias, please adapt the code.")
    return model
