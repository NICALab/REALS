import argparse
import torch
from .model import REALS_model
import time


def REALS(data: torch.Tensor, args: argparse.Namespace, data_shape):
    lr_wwt = args.lr_wwt
    lr_tau = args.lr_tau
    epoch = args.epoch
    batch_size = args.batch_size
    device = args.device
    rank = args.rank
    tau = args.tau
    verbose = args.verbose

    model = REALS_model(data_shape, tau, device, k=rank).to(device)

    if tau == 'euclidean':
        optim = torch.optim.Adam([
            {'params': model.parameters(), 'lr': lr_wwt},
            {'params': model.theta_ro, 'lr': lr_tau},
            {'params': model.theta_tr, 'lr': lr_tau}
        ], weight_decay=0)
    else:
        optim = torch.optim.Adam([
            {'params': model.parameters(), 'lr': lr_wwt},
            {'params': model.theta, 'lr': lr_tau}
        ], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=600, gamma=0.1)
    batch_iter = data.size(0) // batch_size

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_ev.record()
    # Train
    total_loss = None
    for i in range(epoch):
        total_loss = 0.0
        for _ in range(batch_iter):
            batch = data.to(device)
            L, L_stat = model(batch)
            S = L - L_stat
            loss = torch.norm(S, p=1)
            optim.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optim.step()
            model.clamp_theta()
        scheduler.step()

        if epoch > 4:
            if (i % (epoch // 5) == 0 or i == epoch - 1) and verbose:
                print(f"""[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] EPOCH : [{i}/{epoch}] : {(total_loss):.4f}""")

    # Inference
    test_split = data.split(batch_size, dim=0)
    del data

    L_s, S_s, L_stat_s = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_split):
            batch = batch.to(device)
            L, L_stat = model(batch)
            S = L - L_stat
            L = L.to("cpu")
            S = S.to("cpu")
            L_stat = L_stat.to("cpu")
            L_s.append(L)
            S_s.append(S)
            L_stat_s.append(L_stat)

        total_L = torch.cat(L_s, dim=0)
        total_S = torch.cat(S_s, dim=0)
        total_L_stat = torch.cat(L_stat_s, dim=0)

    end_ev.record()
    torch.cuda.synchronize()

    del batch, L, S

    if total_loss == None:
        total_loss = 0.0

    return total_L, total_S, total_L_stat, total_loss, model, start_ev.elapsed_time(end_ev)