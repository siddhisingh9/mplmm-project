import torch
from torch import nn
from torch.utils.data import DataLoader
from src import model as mm
from src.utils import *
import torch.optim as optim
import time
import numpy as np
from types import SimpleNamespace
from torch.optim.lr_scheduler import ReduceLROnPlateau


from src.eval_metrics import *


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.pretrained_model is not None:
        model = getattr(mm, "PromptModel")(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model)
    else:
        model = getattr(mm, "MULTModel")(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]
    selection_metric = getattr(hyp_params, "selection_metric", "loss")
    selection_modes = range(6)

    def build_selection_loader(split="valid"):
        args = SimpleNamespace(
            dataset=hyp_params.dataset,
            data_path=hyp_params.data_path,
            drop_rate=0.0,
            iemo_audio_len=getattr(hyp_params, "iemo_audio_len", 350),
            sims_norm=getattr(hyp_params, "sims_norm", 1),
        )
        dataset = get_data(args, split=split, full_data=True)
        loader_kwargs = {
            "batch_size": hyp_params.batch_size,
            "shuffle": False,
        }
        if hyp_params.dataset == "iemocap":
            loader_kwargs["collate_fn"] = dataset.collate_fn
        return DataLoader(dataset, **loader_kwargs)

    def evaluate_selection_metrics(model, loader):
        model.eval()
        per_mode = []

        with torch.no_grad():
            for mode in selection_modes:
                results = []
                truths = []
                for batch_X, batch_Y, _ in loader:
                    text, audio, vision = batch_X
                    eval_attr = batch_Y.squeeze(dim=-1)

                    if hyp_params.use_cuda:
                        with torch.cuda.device(0):
                            text, audio, vision, eval_attr = (
                                text.cuda(),
                                audio.cuda(),
                                vision.cuda(),
                                eval_attr.cuda(),
                            )
                            if hyp_params.dataset == "iemocap":
                                eval_attr = eval_attr.long()

                    missing_mod = torch.full(
                        (text.size(0),),
                        mode,
                        dtype=torch.long,
                        device=text.device,
                    )

                    batch_size = text.size(0)
                    net = nn.DataParallel(model) if batch_size > 10 else model
                    preds = net(text, audio, vision, missing_mod)

                    if hyp_params.dataset == "iemocap":
                        preds = preds.view(-1, 4)
                        eval_attr = eval_attr.view(-1)

                    results.append(preds.detach().cpu())
                    truths.append(eval_attr.detach().cpu())

                results = torch.cat(results)
                truths = torch.cat(truths)

                if hyp_params.dataset == "iemocap":
                    predicted_labels = results.argmax(dim=1).numpy()
                    true_labels = truths.view(-1).long().numpy()
                else:
                    predicted_labels = results.view(-1).numpy() >= 0
                    true_labels = truths.view(-1).numpy() >= 0

                per_mode.append(
                    {
                        "acc": accuracy_score(true_labels, predicted_labels) * 100,
                        "f1": f1_score(
                            true_labels,
                            predicted_labels,
                            average="weighted",
                            zero_division=0,
                        )
                        * 100,
                    }
                )

        return {
            "avg_acc": float(np.mean([row["acc"] for row in per_mode])),
            "avg_f1": float(np.mean([row["f1"] for row in per_mode])),
        }

    selection_loader = None
    if selection_metric != "loss":
        selection_loader = build_selection_loader(split="valid")

    def train(model, optimizer, criterion):
        model.train()
        num_batches = len(train_loader)
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(),
                    )
                    if hyp_params.dataset == "iemocap":
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(text, audio, vision, missing_mod)

            if hyp_params.dataset == "iemocap":
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
            assert hyp_params.dataset == "iemocap" or preds.shape == eval_attr.shape, (
                f"preds {tuple(preds.shape)} vs labels {tuple(eval_attr.shape)}: "
                "loss would broadcast"
            )
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(loader):
                text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = (
                            text.cuda(),
                            audio.cuda(),
                            vision.cuda(),
                            eval_attr.cuda(),
                        )
                        if hyp_params.dataset == "iemocap":
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds = net(text, audio, vision, missing_mod)
                if hyp_params.dataset == "iemocap":
                    preds = preds.view(-1, 4)
                    eval_attr = eval_attr.view(-1)
                assert hyp_params.dataset == "iemocap" or preds.shape == eval_attr.shape, (
                    f"preds {tuple(preds.shape)} vs labels {tuple(eval_attr.shape)}: "
                    "loss would broadcast"
                )
                total_loss += criterion(preds, eval_attr).item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    best_selection_score = -1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        selection_results = None
        if selection_loader is not None:
            selection_results = evaluate_selection_metrics(model, selection_loader)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        print("-" * 50)
        summary = (
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, val_loss, test_loss
            )
        )
        if selection_results is not None:
            summary += " | Avg Valid ACC {:5.2f} | Avg Valid F1 {:5.2f}".format(
                selection_results["avg_acc"],
                selection_results["avg_f1"],
            )
        print(summary)
        print("-" * 50)

        should_save = val_loss < best_valid
        if selection_results is not None:
            current_selection_score = selection_results[selection_metric]
            should_save = (
                current_selection_score > best_selection_score
                or (
                    np.isclose(current_selection_score, best_selection_score)
                    and val_loss < best_valid
                )
            )

        if should_save:
            print(f"Saved model at {hyp_params.name}")
            torch.save(model, hyp_params.name)
            best_valid = val_loss
            if selection_results is not None:
                best_selection_score = selection_results[selection_metric]

    model = torch.load(hyp_params.name, weights_only=False)
    _, results, truths = evaluate(model, criterion, test=True)

    if hyp_params.dataset == "mosei":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == "mosi":
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == "iemocap":
        eval_iemocap(results, truths)
    elif hyp_params.dataset == "sims":
        eval_sims(results, truths)
