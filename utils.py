def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    rate = [1, 0.1, 1, 1]
    if epoch < 1:
        print(rate)
    for i in range(len(rate)):
        optimizer.param_groups[i]['lr'] = init_lr * decay * rate[i]
