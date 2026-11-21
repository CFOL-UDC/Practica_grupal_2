"""Simulador de clientes maliciosos para el reto (no entra en el envío).

reto.py tiene los ganchos ``poison`` (etiquetas) y ``attack`` (lo que se sube) pero no los usa.
Aquí se implementan los ataques que describe el enunciado para probar las defensas:

    python scripts/reto_ataques.py --seeds 0 1 2 --ataque mezcla
    python scripts/reto_ataques.py --ataque ninguno --set server.aggregator=fedavg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reto import Scenario, Session, StrategyError, run  # noqa: E402

ATAQUES = ["ninguno", "permutar", "desplazar", "aleatorias", "invertir", "escalar", "ruido", "mezcla"]
ETIQUETAS = {"permutar", "desplazar", "aleatorias"}
ENVIO = {"invertir", "escalar", "ruido"}


def hacer_ganchos(ataque, malos, n_clases, seed, tocar_enteros=False, escala=10.0):
    rng = np.random.default_rng(1000 + seed)
    tipo = {}
    for cid in malos:
        if ataque == "mezcla":
            tipo[cid] = (rng.choice(["", "permutar", "desplazar", "aleatorias"]), rng.choice(["", "invertir", "escalar", "ruido"]))
        else:
            tipo[cid] = (ataque if ataque in ETIQUETAS else "", ataque if ataque in ENVIO else "")
    perms = {cid: torch.from_numpy(rng.permutation(n_clases)) for cid in malos}
    gen = torch.Generator().manual_seed(seed)

    def poison(cid, y):
        t = tipo.get(cid, ("", ""))[0]
        if t == "permutar":
            return perms[cid][y]
        if t == "desplazar":
            return (y + 1) % n_clases
        if t == "aleatorias":
            return torch.randint(0, n_clases, y.shape, generator=gen)
        return y

    def attack(cid, payload, global_state):
        t = tipo.get(cid, ("", ""))[1]
        if not t:
            return payload
        out = {}
        for k, v in payload.items():
            if not v.is_floating_point() and not tocar_enteros:
                out[k] = v
                continue
            vf = v.float()
            g = global_state.get(k)
            base = g.float() if g is not None and g.shape == v.shape else torch.zeros_like(vf)
            d = vf - base
            if t == "invertir":
                r = base - d
            elif t == "escalar":
                r = base + escala * d
            else:
                sigma = d.std() if d.numel() > 1 else d.abs().mean()  # con un solo valor, std() no existe
                r = vf + sigma * torch.randn(vf.shape, generator=gen)
            out[k] = r.to(v.dtype) if v.is_floating_point() else r.round().clamp(-128, 127).to(v.dtype) if v.dtype == torch.int8 else r.to(v.dtype)
        return out

    return poison, attack, tipo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reto.yaml")
    ap.add_argument("--set", nargs="+", action="extend", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ataque", default="mezcla", choices=ATAQUES)
    ap.add_argument("--fraccion", type=float, default=0.2)
    ap.add_argument("--enteros", action="store_true", help="los ataques también tocan tensores enteros")
    ap.add_argument("--escala", type=float, default=10.0)
    ap.add_argument("--data", default="data/reto")
    args = ap.parse_args()

    x = torch.from_numpy(np.load(Path(args.data) / "train_x.npy"))
    y = torch.from_numpy(np.load(Path(args.data) / "train_y.npy")).long()
    perm = np.random.default_rng(12345).permutation(len(x))
    n_val = len(x) // 10
    xv, yv, xt, yt = x[perm[:n_val]], y[perm[:n_val]], x[perm[n_val:]], y[perm[n_val:]]

    accs = []
    for seed in args.seeds:
        malos = sorted(np.random.default_rng(500 + seed).choice(100, int(args.fraccion * 100), replace=False).tolist())
        if args.ataque == "ninguno":
            malos = []
        poison, attack, _ = hacer_ganchos(args.ataque, malos, 100, seed, args.enteros, args.escala)
        session = Session(Path.cwd(), args.config, args.set)
        try:
            state, stats = run(session, Scenario(seed=seed), xt, yt, poison=poison, attack=attack)
        except StrategyError as e:
            raise SystemExit(f"Error: {e}")
        acc = (session.predict(state, xv) == yv).float().mean().item()
        accs.append(acc)
        extra = getattr(session.aggregator, "resumen", lambda: "")()
        print(f"semilla {seed} | ataque {args.ataque} | acc {acc:.4f} | {stats.time_s:.1f} s {extra}")
    print(json.dumps({"ataque": args.ataque, "media": round(float(np.mean(accs)), 4), "desv": round(float(np.std(accs)), 4)}))


if __name__ == "__main__":
    main()
