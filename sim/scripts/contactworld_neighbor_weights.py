"""Train/evaluate observation-conditioned convex mixtures of demonstration actions.

No target action enters the weight network. Candidate actions are read only
after retrieval. Every training query excludes its entire source episode.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn

from contactworld_neighbor_features import MODES, choose_device, save_npz


def scale_features(data, fit, mode, stats=None):
    columns, fitted, slices = [], {}, []
    for key in MODES[mode]:
        x = np.asarray(data[key], np.float32)
        if stats is None:
            mean = x[fit].mean(0)
            std = x[fit].std(0)
            floor = max(float(np.sqrt(np.mean(std**2)))*.1, 1e-5)
            std = np.maximum(std, floor)
        else:
            mean, std = stats[key]
        fitted[key] = (mean, std)
        # Equal expected block distances regardless of feature dimensionality.
        columns.append(((x-mean)/std)/np.sqrt(x.shape[1]))
        slices.append((sum(c.shape[1] for c in columns[:-1]), sum(c.shape[1] for c in columns)))
    return np.concatenate(columns, axis=1).astype(np.float32), fitted, slices


def retrieve(x, source, queries, library, k=10):
    """One nearest frame per allowed source; prohibit the query source."""
    episodes = np.unique(source[library])
    if len(episodes) <= k:
        raise ValueError(f"Need at least {k+1} candidate source episodes, got {len(episodes)}")
    indexes, distances = [], []
    for start in range(0, len(queries), 128):
        q = queries[start:start+128]
        distance = cdist(x[q], x[library], "sqeuclidean") / max(1, x.shape[1])
        best_index, best_distance = [], []
        for ep in episodes:
            mask = np.flatnonzero(source[library] == ep)
            block = distance[:, mask]
            nearest = block.argmin(1)
            d = block[np.arange(len(q)), nearest]
            d[source[q] == ep] = np.inf
            best_index.append(library[mask[nearest]])
            best_distance.append(d)
        bi, bd = np.stack(best_index, 1), np.stack(best_distance, 1)
        selected = np.argsort(bd, axis=1, kind="stable")[:, :k]
        indexes.append(np.take_along_axis(bi, selected, axis=1))
        distances.append(np.take_along_axis(bd, selected, axis=1))
    ix, dist = np.concatenate(indexes), np.concatenate(distances).astype(np.float32)
    assert not np.any(source[ix] == source[queries, None])
    assert np.isfinite(dist).all()
    return ix, dist


def prior_logits(dist, k, temperature):
    d = dist[:, :k]
    # Common scale from top ten, so changing k does not redefine temperature.
    scale = np.maximum(dist[:, -1]-dist[:, 0], np.maximum(dist[:, 0]*.05, 1e-8))
    return -(d-d[:, :1])/scale[:, None]/temperature


def softmax(x):
    y = np.exp(x-x.max(1, keepdims=True))
    return y/y.sum(1, keepdims=True)


class WeightNet(nn.Module):
    def __init__(self, features, blocks):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features*3+blocks, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query, candidates, logits, slices):
        q = query[:, None].expand_as(candidates)
        delta = q-candidates
        blocks = torch.stack([(delta[..., a:b]**2).sum(-1) for a, b in slices], -1)
        residual = 2*torch.tanh(self.net(torch.cat([q, candidates, delta.abs(), blocks], -1)).squeeze(-1))
        return torch.softmax(logits+residual, dim=1)


def loss_per_frame(pred, target, action_scale):
    error = ((pred-target)/action_scale[None, None])**2
    return .75*error[:, 0].mean(1)+.25*error[:, 1:].mean((1, 2))


def episode_mean(values, sources):
    return float(np.mean([values[sources == s].mean() for s in np.unique(sources)]))


def fit_network(x, actions, source, train, train_nn, train_dist, slices, k, temperature,
                action_scale, epochs, device, seed, validation=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WeightNet(x.shape[1], len(slices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    xt = torch.as_tensor(x, device=device)
    at = torch.as_tensor(actions, device=device)
    st = torch.as_tensor(action_scale, device=device)
    nn_ix = torch.as_tensor(train_nn[:, :k], device=device)
    logits = torch.as_tensor(prior_logits(train_dist, k, temperature), device=device)
    qt = torch.as_tensor(train, device=device)
    # Episode-balanced sampling prevents long demonstrations dominating.
    counts = {s: np.count_nonzero(source[train] == s) for s in np.unique(source[train])}
    probability = np.array([1/counts[s] for s in source[train]])
    probability /= probability.sum()
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch, best_loss = 0, float("inf")
    history = []
    started = time.monotonic()
    # Include epoch zero, the exact similarity-initialized network, as a
    # selectable outcome. A learned model need not beat its initial prior.
    for epoch in range(epochs+1):
        if epoch:
            model.train()
            sampled = rng.choice(len(train), size=len(train), replace=True, p=probability)
            train_losses = []
            for start in range(0, len(sampled), 128):
                b = torch.as_tensor(sampled[start:start+128], device=device)
                weights = model(xt[qt[b]], xt[nn_ix[b]], logits[b], slices)
                prediction = torch.einsum("bk,bkhd->bhd", weights, at[nn_ix[b]])
                mse = .75*(((prediction[:, 0]-at[qt[b], 0])/st)**2).mean()
                mse += .25*(((prediction[:, 1:]-at[qt[b], 1:])/st)**2).mean()
                prior = logits[b].softmax(-1)
                kl = (weights*(weights.clamp_min(1e-8).log()-prior.clamp_min(1e-8).log())).sum(1).mean()
                objective = mse+.01*kl
                optimizer.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                train_losses.append(float(objective.detach().cpu()))
        if validation is not None:
            vq, vi, vd = validation
            predicted, _ = infer(model, x, actions, vq, vi, vd, slices, k, temperature, device)
            score = episode_mean(loss_per_frame(predicted, actions[vq], action_scale), source[vq])
        else:
            score = 0.
        history.append({"epoch": epoch, "validation_loss": score if validation is not None else None})
        if epoch % 5 == 0 or epoch == epochs:
            print(f"weights epoch={epoch}/{epochs} validation={score if validation is not None else 'refit'} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if validation is None or score < best_loss:
            best_loss, best_epoch = score, epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, best_epoch, history


@torch.inference_mode()
def infer(model, x, actions, query, candidates, distances, slices, k, temperature, device):
    model.eval()
    predictions, weights = [], []
    for start in range(0, len(query), 128):
        q, ix = query[start:start+128], candidates[start:start+128, :k]
        logits = prior_logits(distances[start:start+128], k, temperature)
        w = model(torch.as_tensor(x[q], device=device), torch.as_tensor(x[ix], device=device),
                  torch.as_tensor(logits, device=device), slices).cpu().numpy()
        predictions.append(np.einsum("bk,bkhd->bhd", w, actions[ix]))
        weights.append(w)
    return np.concatenate(predictions), np.concatenate(weights)


def predict_baseline(actions, ix, dist, k, temperature):
    w = np.ones((len(ix), k), np.float32)/k if temperature == 0 else softmax(prior_logits(dist, k, temperature))
    return np.einsum("bk,bkhd->bhd", w, actions[ix[:, :k]]), w


def select_prior(actions, source, query, ix, dist, action_scale):
    scores = []
    for k in (1, 3, 5, 10):
        for temperature in (0., .5, 1., 2.):
            p, _ = predict_baseline(actions, ix, dist, k, temperature)
            value = episode_mean(loss_per_frame(p, actions[query], action_scale), source[query])
            scores.append({"k": k, "temperature": temperature, "validation_loss": value})
    return min(scores, key=lambda s: s["validation_loss"]), scores


def metrics(pred, target):
    diff = pred-target
    dot = np.sum(pred[:, 0, :3]*target[:, 0, :3], axis=-1)
    den = np.linalg.norm(pred[:, 0, :3], axis=-1)*np.linalg.norm(target[:, 0, :3], axis=-1)
    cos = np.divide(dot, den, out=np.full(len(pred), np.nan), where=den > .01)
    return {"translation_mm": np.linalg.norm(diff[:, 0, :3], axis=-1)*10,
            "xy_mm": np.linalg.norm(diff[:, 0, :2], axis=-1)*10,
            "rotation_deg": np.linalg.norm(diff[:, 0, 3:], axis=-1)*(.05*180/np.pi),
            "chunk_translation_mm": np.sqrt(np.mean(np.sum(diff[:, :, :3]**2, axis=-1), axis=1))*10,
            "chunk_rotation_deg": np.sqrt(np.mean(np.sum(diff[:, :, 3:]**2, axis=-1), axis=1))*(.05*180/np.pi),
            "cos3": cos}


def uncertainty(pred, weights, candidate_actions):
    residual = candidate_actions-pred[:, None]
    variance = np.einsum("bk,bkhd->bhd", weights, residual**2)
    unit = candidate_actions[:, :, 0, :2]
    norms = np.linalg.norm(unit, axis=-1)
    cos = np.einsum("bki,bli->bkl", unit, unit)/np.maximum(norms[:, :, None]*norms[:, None, :], 1e-8)
    opposite = (cos < -.5) & (norms[:, :, None] > .1) & (norms[:, None, :] > .1)
    return {"variance": variance, "effective_k": 1/(weights**2).sum(1),
            "opposing_xy_mass": np.einsum("bk,bl,bkl->b", weights, weights, opposite),
            "spread_translation_mm": np.sqrt(variance[:, 0, :3].sum(1))*10}


def save_bundle(path, model, scaler, slices, k, temperature, mode, action_scale, library, epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    temp = path.with_suffix(".tmp.pt")
    torch.save({"state_dict": state, "scaler": scaler, "slices": slices, "k": k,
                "temperature": temperature, "mode": mode, "action_scale": action_scale,
                "library": library, "epochs": epochs}, temp)
    temp.replace(path)


def train_cv(data, output, modes, epochs=30, folds=5, seed=20261003, device="auto", outer_limit=None):
    output = Path(output)
    device = choose_device(device)
    source, actions = data["source"], data["action"]
    episodes = np.random.default_rng(seed).permutation(np.unique(source))
    groups = np.array_split(episodes, folds)
    split_path = output / "splits.json"
    splits = {str(i): {"outer_test_sources": test.tolist(), "outer_train_sources": np.setdiff1d(episodes, test).tolist()}
              for i, test in enumerate(groups)}
    output.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(splits, indent=2)+"\n")
    for mode in modes:
        for fold, test_sources in enumerate(groups):
            if outer_limit is not None and fold >= outer_limit:
                continue
            folder = output / mode / f"fold_{fold}"
            if (folder / "complete.json").exists():
                print(f"resume: {mode} fold {fold} complete", flush=True)
                continue
            print(f"START {mode} fold {fold}: grouped inner validation then outer refit", flush=True)
            outer_sources = np.setdiff1d(episodes, test_sources)
            shuffled = np.random.default_rng(seed+fold).permutation(outer_sources)
            val_sources = shuffled[:max(3, int(len(shuffled)*.2))]
            fit_sources = np.setdiff1d(outer_sources, val_sources)
            fit, val, test = [np.flatnonzero(np.isin(source, s)) for s in (fit_sources, val_sources, test_sources)]
            x, _, slices = scale_features(data, fit, mode)
            fit_nn, fit_d = retrieve(x, source, fit, fit)
            val_nn, val_d = retrieve(x, source, val, fit)
            action_scale = np.maximum(actions[fit].std((0, 1)), .05)
            prior, grid = select_prior(actions, source, val, val_nn, val_d, action_scale)
            # Weight learning always gets ten candidates: otherwise a selected
            # k=1 baseline leaves the network nothing to learn. Tune its prior
            # temperature on the same validation split, then compare to the
            # independently selected best fixed baseline.
            learned_prior = min((g for g in grid if g["k"] == 10 and g["temperature"] > 0), key=lambda g: g["validation_loss"])
            temperature = learned_prior["temperature"]
            model, best_epoch, history = fit_network(x, actions, source, fit, fit_nn, fit_d, slices,
                10, temperature, action_scale, epochs, device, seed+fold,
                validation=(val, val_nn, val_d))
            del model
            outer = np.flatnonzero(np.isin(source, outer_sources))
            x, scaler, slices = scale_features(data, outer, mode)
            nn_train, d_train = retrieve(x, source, outer, outer)
            nn_test, d_test = retrieve(x, source, test, outer)
            action_scale = np.maximum(actions[outer].std((0, 1)), .05)
            model, _, _ = fit_network(x, actions, source, outer, nn_train, d_train, slices,
                10, temperature, action_scale, best_epoch, device, seed+fold)
            learned, w = infer(model, x, actions, test, nn_test, d_test, slices, 10, temperature, device)
            fixed, fw = predict_baseline(actions, nn_test, d_test, prior["k"], prior["temperature"])
            initialized, iw = predict_baseline(actions, nn_test, d_test, 10, temperature)
            nearest = actions[nn_test[:, 0]]
            global_mean = np.mean([actions[source == s].mean(0) for s in outer_sources], axis=0)
            save_npz(folder / "predictions.npz", query=test, source=source[test], frame=data["frame"][test],
                phase=data["phase"][test], target=actions[test], learned=learned, fixed=fixed,
                initialized=initialized, nearest=nearest, mean=np.broadcast_to(global_mean, learned.shape),
                neighbors=nn_test, neighbor_sources=source[nn_test], neighbor_frames=data["frame"][nn_test],
                distances=d_test, weights=w, fixed_weights=fw, initialized_weights=iw,
                **uncertainty(learned, w, actions[nn_test]))
            save_bundle(folder / "model.pt", model, scaler, slices, 10, temperature, mode, action_scale, outer, best_epoch)
            info = {"fold": fold, "mode": mode, "best_epoch": best_epoch, "fixed_prior": prior,
                    "learned_prior": learned_prior, "inner_fit_sources": fit_sources.tolist(),
                    "inner_validation_sources": val_sources.tolist(), "outer_test_sources": test_sources.tolist(),
                    "history": history, "grid": grid}
            (folder / "complete.json").write_text(json.dumps(info, indent=2)+"\n")
            print(f"complete {mode} fold {fold}: epochs={best_epoch} fixed={prior}", flush=True)
        folder = output / mode / "final"
        if (folder / "complete.json").exists():
            continue
        print(f"START {mode} final fit on all train demonstrations", flush=True)
        infos = [json.loads((output/mode/f"fold_{i}/complete.json").read_text()) for i in range(outer_limit or folds)]
        final_epochs = int(np.median([i["best_epoch"] for i in infos]))
        temperature = float(np.median([i["learned_prior"]["temperature"] for i in infos]))
        library = np.arange(len(source))
        x, scaler, slices = scale_features(data, library, mode)
        ix, d = retrieve(x, source, library, library)
        scale = np.maximum(actions.std((0, 1)), .05)
        model, _, _ = fit_network(x, actions, source, library, ix, d, slices, 10, temperature,
                                  scale, final_epochs, device, seed)
        save_bundle(folder / "model.pt", model, scaler, slices, 10, temperature, mode, scale, library, final_epochs)
        (folder / "complete.json").write_text(json.dumps({"epochs": final_epochs, "temperature": temperature})+"\n")


def load_predictor(path, device):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    width = sum(len(bundle["scaler"][k][0]) for k in MODES[bundle["mode"]])
    model = WeightNet(width, len(bundle["slices"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    return model, bundle


def predict_observations(train, observations, model_path, device="cpu"):
    model, bundle = load_predictor(model_path, device)
    n = len(train["source"])
    combined = {k: np.concatenate([train[k], observations[k]]) for k in MODES[bundle["mode"]]}
    # Negative query IDs cannot overlap a demonstration episode.
    source = np.r_[train["source"], -np.ones(len(observations["state"]), int)]
    x, _, _ = scale_features(combined, bundle["library"], bundle["mode"], bundle["scaler"])
    query = np.arange(n, len(source))
    ix, d = retrieve(x, source, query, bundle["library"])
    pred, w = infer(model, x, train["action"], query, ix, d, bundle["slices"],
                    bundle["k"], bundle["temperature"], device)
    prior, _ = predict_baseline(train["action"], ix, d, bundle["k"], bundle["temperature"])
    return {"learned": pred, "initialized": prior, "nearest": train["action"][ix[:, 0]],
            "neighbors": ix, "neighbor_sources": train["source"][ix], "neighbor_frames": train["frame"][ix],
            "distances": d, "weights": w, **uncertainty(pred, w, train["action"][ix])}
