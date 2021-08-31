import argparse
import time
import torch
from .model import REALS_model_minibatch


def REALS_minibatch(data: torch.Tensor, args: argparse.Namespace, data_shape):
    lr_wwt = args.lr_wwt
    lr_tau = args.lr_tau
    epoch = args.epoch
    batch_size = args.batch_size
    device = args.device
    rank = args.rank
    tau = args.tau
    verbose = args.verbose

    model = REALS_model_minibatch(data_shape, tau, device, k=rank).to(device)

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

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_ev.record()

    # simple dataset
    from torch.utils.data import Dataset, DataLoader

    class SimpleDataset(Dataset):
        def __init__(self, t):
            self.t = t

        def __len__(self):
            return self.t

        def __getitem__(self, idx):
            return torch.LongTensor([idx])

    dataset = SimpleDataset(data_shape[-1])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Train
    total_loss = None
    for i in range(epoch):
        total_loss = 0.0
        for batch_idx, r_index in enumerate(train_dataloader):
            r_index = r_index.flatten().to(device)
            batch = data[r_index, :].to(device)
            L, L_stat = model(batch, r_index)
            S = L - L_stat
            loss = torch.norm(S, p=1)
            optim.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optim.step()
            model.clamp_theta()

        if epoch > 4:
            if (i % (epoch // 5) == 0 or i == epoch - 1) and verbose:
                print(f"""[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] EPOCH : [{i}/{epoch}] : {(total_loss):.4f}""")

    # Inference
    L_s, S_s, L_stat_s = [], [], []
    with torch.no_grad():
        for batch_idx, r_index in enumerate(test_dataloader):
            r_index = r_index.flatten().to(device)
            batch = data[r_index, :].to(device)
            L, L_stat = model(batch, r_index)
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