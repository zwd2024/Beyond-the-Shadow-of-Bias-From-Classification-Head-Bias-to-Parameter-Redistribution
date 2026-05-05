import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def TS_BGRM(
    args,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    model,
    device,
):
    for param in model.parameters():
        param.requires_grad = False

    # 只解冻最后一层（分类器层） [Only thaw the last layer (classifier layer)]
    # 如果你用的是其他模型，请自行替换成你模型的最后一层名字
    # If you are using another model, please replace it with the name of the last layer of your model by yourself
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True

    # 优化器(optimizer)
    ft_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    model.train()

    # 获取分类头(Get the category header)
    if hasattr(model, 'fc'):
        cls_layer = model.fc
    elif hasattr(model, 'classifier'):
        cls_layer = model.classifier
    else:
        raise RuntimeError("Model has no fc or classifier")

    num_classes = cls_layer.out_features
    forget_classes = set(args.unlearn_class)

    # 第一阶段（phase one）
    for _ in range(args.unlearning_epochs):
        forget_iter = iter(forget_loader)
        while True:
            try:
                x_f, y_f = next(forget_iter)
            except StopIteration:
                break

            x_f, y_f = x_f.to(device), y_f.to(device)
            ft_optimizer.zero_grad()

            # 前向(forward propagation)
            logits = model(x_f)

            # 正常 loss（遗忘类方向） [Normal loss (Forgetting type direction)]
            loss = -1.0 * nn.CrossEntropyLoss()(logits, y_f)

            # 反向(backpropagation)
            loss.backward()

            # 反转“非遗忘类”分类头梯度(Reverse the gradient of the "non-forgetting" category header)
            W = cls_layer.weight
            b = cls_layer.bias if cls_layer.bias is not None else None

            for c in range(num_classes):
                if c not in forget_classes:
                    if W.grad is not None:
                        W.grad[c] *= 1.0
                    if b is not None and b.grad is not None:
                        b.grad[c] *= args.unlearning_loss * 1.0

            ft_optimizer.step()

    # 第二阶段(phase Two)
    fine_tune_epochs = args.ft_epoch
    for _ in range(fine_tune_epochs):
        retain_iter = iter(retain_loader)
        while True:
            try:
                x_r, y_r = next(retain_iter)
            except StopIteration:
                break

            x_r, y_r = x_r.to(device), y_r.to(device)
            ft_optimizer.zero_grad()

            logits = model(x_r)
            loss = nn.CrossEntropyLoss()(logits, y_r)
            loss.backward()
            ft_optimizer.step()

    return model