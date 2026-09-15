from pprint import pprint
from sklearn.svm import LinearSVC
from math import log, pi
import os
import torch
import torch.distributed as dist
import random
import numpy as np


def modify_one_attribute(attributes:torch.Tensor,idx:int=-1,scale:float=None,mode='random'):
    """

    Args:
        attributes (torch.Tensor): 0-1 tensor
        idx (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """    
    attributes = attributes.to(torch.float32)
    bs,columns = attributes.shape[:2]
    new_attributes = attributes.detach().clone()
    
    if mode == 'keep':
        return None,new_attributes
    
    if idx < 0 or idx >= columns:
        # if modified index is not given , randomly generate one !
        if mode=='random':
            idx = torch.randint(0,columns,(bs,))
            new_attributes[torch.arange(bs), idx] = torch.randint(0, 2, (bs,),dtype=torch.float32).to(new_attributes)
        elif mode == 'uniform':
            idx = torch.randint(0,columns,(bs,))
            new_attributes[torch.arange(bs), idx] = torch.rand((bs,),dtype=torch.float32).to(new_attributes)
        elif mode=='negative':
            idx = torch.randint(0,columns,(1,))
            new_attributes[torch.arange(bs), idx] = 1. -new_attributes[torch.arange(bs), idx]
        
    elif scale is not None:
        new_attributes[attributes[:,idx]==1,idx] = 1.0 - scale
        new_attributes[attributes[:,idx]==0,idx] = scale
    else:
        new_attributes[:,idx] = 1 - new_attributes[:,idx]
    
    return idx,new_attributes


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


# Visualization


# Augmentation


def validate_classification(loaders, model, args):
    train_loader, test_loader = loaders

    def _make_iter_(loader):
        iterator = iter(loader)
        return iterator

    tr_latent = []
    tr_label = []
    for data in _make_iter_(train_loader):
        tr_pc = data['train_points']
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
        latent = model.encode(tr_pc)
        label = data['cate_idx']
        tr_latent.append(latent.cpu().detach().numpy())
        tr_label.append(label.cpu().detach().numpy())
    tr_label = np.concatenate(tr_label)
    tr_latent = np.concatenate(tr_latent)

    te_latent = []
    te_label = []
    for data in _make_iter_(test_loader):
        tr_pc = data['train_points']
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
        latent = model.encode(tr_pc)
        label = data['cate_idx']
        te_latent.append(latent.cpu().detach().numpy())
        te_label.append(label.cpu().detach().numpy())
    te_label = np.concatenate(te_label)
    te_latent = np.concatenate(te_latent)

    clf = LinearSVC(random_state=0)
    clf.fit(tr_latent, tr_label)
    test_pred = clf.predict(te_latent)
    test_gt = te_label.flatten()
    acc = np.mean((test_pred == test_gt).astype(float)) * 100.
    res = {'acc': acc}
    print("Acc:%s" % acc)
    return res


def validate_conditioned(loader, model, args, max_samples=None, save_dir=None):
    from metrics.evaluation_metrics import EMD_CD
    all_idx = []
    all_sample = []
    all_ref = []
    ttl_samples = 0
    iterator = iter(loader)

    for data in iterator:
        # idx_b, tr_pc, te_pc = data[:3]
        idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)

        if tr_pc.size(1) > te_pc.size(1):
            tr_pc = tr_pc[:, :te_pc.size(1), :]
        out_pc = model.reconstruct(tr_pc, num_points=te_pc.size(1))

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
        all_idx.append(idx_b)

        ttl_samples += int(te_pc.size(0))
        if max_samples is not None and ttl_samples >= max_samples:
            break

    # Compute MMD and CD
    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("[rank %s] Recon Sample size:%s Ref size: %s" % (args.rank, sample_pcs.size(), ref_pcs.size()))

    if save_dir is not None and args.save_val_results:
        smp_pcs_save_name = os.path.join(save_dir, "smp_recon_pcls_gpu%s.npy" % args.gpu)
        ref_pcs_save_name = os.path.join(save_dir, "ref_recon_pcls_gpu%s.npy" % args.gpu)
        np.save(smp_pcs_save_name, sample_pcs.cpu().detach().numpy())
        np.save(ref_pcs_save_name, ref_pcs.cpu().detach().numpy())
        print("Saving file:%s %s" % (smp_pcs_save_name, ref_pcs_save_name))

    res = EMD_CD(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    mmd_cd = res['MMD-CD'] if 'MMD-CD' in res else None
    mmd_emd = res['MMD-EMD'] if 'MMD-EMD' in res else None

    print("MMD-CD  :%s" % mmd_cd)
    print("MMD-EMD :%s" % mmd_emd)

    return res


def validate_sample(loader, model, args, max_samples=None, save_dir=None):
    from metrics.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD
    all_sample = []
    all_ref = []
    ttl_samples = 0

    iterator = iter(loader)

    for data in iterator:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        _, out_pc = model.sample(te_pc.size(0), te_pc.size(1), gpu=args.gpu)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

        ttl_samples += int(te_pc.size(0))
        if max_samples is not None and ttl_samples >= max_samples:
            break

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("[rank %s] Generation Sample size:%s Ref size: %s"
          % (args.rank, sample_pcs.size(), ref_pcs.size()))

    if save_dir is not None and args.save_val_results:
        smp_pcs_save_name = os.path.join(save_dir, "smp_syn_pcls_gpu%s.npy" % args.gpu)
        ref_pcs_save_name = os.path.join(save_dir, "ref_syn_pcls_gpu%s.npy" % args.gpu)
        np.save(smp_pcs_save_name, sample_pcs.cpu().detach().numpy())
        np.save(ref_pcs_save_name, ref_pcs.cpu().detach().numpy())
        print("Saving file:%s %s" % (smp_pcs_save_name, ref_pcs_save_name))

    res = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    pprint(res)

    sample_pcs = sample_pcs.cpu().detach().numpy()
    ref_pcs = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcs, ref_pcs)
    jsd = torch.tensor(jsd).cuda() if args.gpu is None else torch.tensor(jsd).cuda(args.gpu)
    res.update({"JSD": jsd})
    print("JSD     :%s" % jsd)
    return res


def save(model, optimizer, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, start_epoch


