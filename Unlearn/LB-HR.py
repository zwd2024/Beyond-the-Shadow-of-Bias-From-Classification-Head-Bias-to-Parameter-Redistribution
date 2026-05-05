import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def LB_HR(
    args,
    retain_loader: DataLoader,
    model,
    device,
):

    model.to(device)
    model.train()

    # forget_classes = [int(x) for x in args.unlearn_class.split(",")]
    forget_classes = args.unlearn_class
    print("forget_classes:", forget_classes)

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

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=args.gamma,
    )

    for epoch in range(args.epochs_or_steps):
        for batch_idx, (data, target) in enumerate(retain_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            logits = model(data)

            # 正常训练 retain 类 (Normal training of retain classes)
            loss = F.cross_entropy(logits, target)


            # Hinge 下界正则(Hinge lower bound regular)：lambda * sum max(0, b_min - b_c)^2
            b = model.fc.bias
            b_min = args.bias_min
            lam = args.f_bias_loss_rate
            reg = 0.0
            for c in forget_classes:
                reg = reg + torch.max(
                    torch.tensor(0.0, device=b.device),
                    b_min - b[c]
                ) ** 2
            loss = loss + lam * reg

            loss.backward()
            optimizer.step()


            if batch_idx % args.log_interval == 0:
                b_val = model.fc.bias[forget_classes[0]].item()
                print(
                    f"Epoch {epoch} Batch {batch_idx} "
                    f"Loss: {loss.item():.4f} "
                    f"b_forget: {b_val:.4f}"
                )

        scheduler.step()

    return model