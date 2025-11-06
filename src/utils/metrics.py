def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()           # Binarize prediction
    intersection = (pred * target).sum()  # TP
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)