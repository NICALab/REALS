import argparse
import time
import torch
from .model import REALS_model_multi


def REALS_multi(data: torch.Tensor, args: argparse.Namespace, data_shape):
    lr_wwt = args.lr_wwt
    lr_tau = args.lr_tau
    epoch = args.epoch
    batch_size = args.batch_size
    device = args.device
    rank = args.rank
    tau = args.tau
    verbose = args.verbose

    model = REALS_model_multi(data_shape, tau, device, k=rank).to(device)

    optim = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_wwt},
        {'params': model.theta, 'lr': lr_tau}
    ], weight_decay=0)

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
            L, L_stat, L_x2, L_stat_x2, L_x4, L_stat_x4, L_x8, L_stat_x8 = model(batch)
            S = L - L_stat
            S_x2 = L_x2 - L_stat_x2
            S_x4 = L_x4 - L_stat_x4
            S_x8 = L_x8 - L_stat_x8
            loss = torch.norm(S, p=1) + 4 * torch.norm(S_x2, p=1) + 16 * torch.norm(S_x4, p=1) + 64 * torch.norm(S_x8, p=1)
            optim.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optim.step()
            model.clamp_theta()

        if epoch > 4:
            if (i % (epoch // 5) == 0 or i == epoch - 1) and verbose:
                print(model.theta[0])
                print(f"""[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] EPOCH : [{i}/{epoch}] : {(total_loss):.4f}""")

    # Inference
    test_split = data.split(batch_size, dim=0)
    del data

    L_s, S_s, L_stat_s = [], [], []
    L_s_x8, S_s_x8, L_stat_s_x8 = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_split):
            batch = batch.to(device)
            L, L_stat, _, _, _, _, L_x8, L_stat_x8 = model(batch)
            S = L - L_stat
            S_x8 = L_x8 - L_stat_x8
            L = L.to("cpu")
            S = S.to("cpu")
            L_stat = L_stat.to("cpu")
            L_x8 = L_x8.to("cpu")
            S_x8 = S_x8.to("cpu")
            L_stat_x8 = L_stat_x8.to("cpu")
            L_s.append(L)
            S_s.append(S)
            L_stat_s.append(L_stat)
            L_s_x8.append(L_x8)
            S_s_x8.append(S_x8)
            L_stat_s_x8.append(L_stat_x8)

        total_L = torch.cat(L_s, dim=0)
        total_S = torch.cat(S_s, dim=0)
        total_L_stat = torch.cat(L_stat_s, dim=0)
        total_L_x8 = torch.cat(L_s_x8, dim=0)
        total_S_x8 = torch.cat(S_s_x8, dim=0)
        total_L_stat_x8 = torch.cat(L_stat_s_x8, dim=0)

    end_ev.record()
    torch.cuda.synchronize()

    del batch, L, S

    if total_loss == None:
        total_loss = 0.0

    return total_L, total_S, total_L_stat, total_loss, model, start_ev.elapsed_time(end_ev), total_L_x8, total_S_x8, total_L_stat_x8