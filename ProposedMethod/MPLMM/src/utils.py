import torch
from src.mosidata import MOSIData
from src.iemodata import IEMOData
from src.simsdata import SIMSData
from torch.utils.data import DataLoader


class opt:
    A_type = "comparE"
    V_type = "denseface"
    L_type = "bert_large"
    norm_method = "trn"
    corpus_name = "IEMOCAP"
    in_mem = False
    cvNo = 1


def get_data(args, split="train", full_data=False):
    if args.dataset == "iemocap":
        if split == "train":
            data = IEMOData(
                opt,
                args.data_path,
                set_name="trn",
                drop_rate=args.drop_rate,
                full_data=full_data,
                audio_len=getattr(args, "iemo_audio_len", 350),
            )
        elif split == "valid":
            data = IEMOData(
                opt,
                args.data_path,
                set_name="val",
                drop_rate=args.drop_rate,
                full_data=full_data,
                audio_len=getattr(args, "iemo_audio_len", 350),
            )
        elif split == "test":
            data = IEMOData(
                opt,
                args.data_path,
                set_name="tst",
                drop_rate=args.drop_rate,
                full_data=full_data,
                audio_len=getattr(args, "iemo_audio_len", 350),
            )
    elif args.dataset == "mosi" or args.dataset == "mosei":
        data = MOSIData(
            args.data_path, split, drop_rate=args.drop_rate, full_data=full_data
        )
    elif args.dataset == "sims":
        data = SIMSData(
            args.data_path,
            split,
            drop_rate=args.drop_rate,
            full_data=full_data,
            normalize=bool(getattr(args, "sims_norm", 1)),
        )
    return data


def get_loader(args):
    dataloaders = {}
    n_nums = []
    if args.dataset == "iemocap":
        for split in ["train", "valid", "test"]:
            dataset = get_data(args, split)
            # IEMOCAP files are ordered by session/dialogue, so labels come in long
            # runs; optionally shuffle the training split. The generator must live on
            # the default device (main.py sets it to CUDA).
            shuffle = split == "train" and bool(getattr(args, "shuffle_train", 0))
            generator = None
            if shuffle:
                generator = torch.Generator(device=torch.get_default_device())
                generator.manual_seed(args.seed)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                drop_last=False,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                generator=generator,
            )
            orig_dims = dataset.get_dim()
            n_nums.append(len(dataset))
            seq_len = dataset.get_seq_len()
    else:
        for split in ["train", "valid", "test"]:
            dataset = get_data(args, split)
            # MOSI/MOSEI pickles are ordered by video, so optionally shuffle the train split.
            shuffle = split == "train" and bool(getattr(args, "shuffle_train", 0))
            generator = None
            if shuffle:
                generator = torch.Generator(device=torch.get_default_device())
                generator.manual_seed(args.seed)
            dataloaders[split] = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator
            )
            orig_dims = dataset.get_dim()
            n_nums.append(len(dataset))
            seq_len = dataset.get_seq_len()
    return dataloaders, orig_dims, n_nums, seq_len


def transfer_model(new_model, pretrained):
    model = torch.load(pretrained, weights_only=False)
    pretrain_dict = model.state_dict()
    new_dict = new_model.state_dict()
    state_dict = {}
    for k, v in pretrain_dict.items():
        if k in new_dict.keys() and k not in [
            "proj_l.weight",
            "proj_a.weight",
            "proj_v.weight",
            "out_layer.weight",
            "out_layer.bias",
        ]:
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    new_dict.update(state_dict)
    new_model.load_state_dict(new_dict)
    for name, param in new_model.named_parameters():
        if name in pretrain_dict.keys() and name not in [
            "proj_l.weight",
            "proj_a.weight",
            "proj_v.weight",
            "out_layer.weight",
            "out_layer.bias",
        ]:
            param.requires_grad = False
    return new_model
