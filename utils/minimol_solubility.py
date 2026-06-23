"""MiniMol + Ridge aqueous solubility predictor with package-local caches."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Union

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from .. import paths


_CACHE_DIR = paths.cache_path("minimol_solubility")
_MODEL_CACHE = _CACHE_DIR / "minimol_solubility_ridge.pkl"
_AQSOLDB_CACHE = _CACHE_DIR / "aqsoldb_train.csv"


def esol(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    clogp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for atom in mol.GetAromaticAtoms())
    ap = aromatic / heavy if heavy else 0.0
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def _apply_scipy_patch() -> None:
    try:
        import numpy as np
        import scipy.sparse._sputils as sparse_utils

        original = sparse_utils.getdtype

        def patched(dtype=None, a=None, default=None):
            if dtype is not None:
                try:
                    if np.dtype(dtype) == np.float16:
                        dtype = np.float32
                except Exception:
                    pass
            return original(dtype, a, default)

        sparse_utils.getdtype = patched
    except Exception:
        pass


def _load_minimol_class():
    try:
        from minimol import Minimol

        return Minimol
    except Exception:
        repo = os.environ.get("THERAPEUTIC_TOOLS_MINIMOL_REPO")
        if repo:
            sys.path.insert(0, str(Path(repo).expanduser()))
            from minimol import Minimol

            return Minimol
        raise


def _train_and_save(minimol_model) -> object:
    import pandas as pd
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not _AQSOLDB_CACHE.exists():
        raise FileNotFoundError(
            f"AqSolDB training data not found at {_AQSOLDB_CACHE}. "
            "Populate cache/minimol_solubility/aqsoldb_train.csv first."
        )

    df = pd.read_csv(_AQSOLDB_CACHE)
    smiles_col = next(column for column in ["Drug", "SMILES", "smiles"] if column in df.columns)
    smiles = df[smiles_col].dropna().tolist()
    labels = df.loc[df[smiles_col].notna(), "Y"].tolist()

    all_embs, all_y = [], []
    for start in range(0, len(smiles), 64):
        batch_smiles = smiles[start : start + 64]
        batch_labels = labels[start : start + 64]
        try:
            embs = minimol_model(batch_smiles)
            for emb, label in zip(embs, batch_labels):
                if isinstance(emb, torch.Tensor):
                    all_embs.append(emb)
                    all_y.append(float(label))
        except Exception:
            for smi, label in zip(batch_smiles, batch_labels):
                try:
                    emb = minimol_model([smi])
                    if emb and isinstance(emb[0], torch.Tensor):
                        all_embs.append(emb[0])
                        all_y.append(float(label))
                except Exception:
                    pass

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(torch.stack(all_embs).numpy(), all_y)
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _MODEL_CACHE.open("wb") as file:
        pickle.dump(model, file)
    return model


class MiniMolSolubility:
    """Aqueous solubility predictor using MiniMol embeddings and Ridge regression."""

    def __init__(self):
        _apply_scipy_patch()
        Minimol = _load_minimol_class()
        import torch

        self._torch = torch
        self._mm = Minimol()
        if _MODEL_CACHE.exists():
            with _MODEL_CACHE.open("rb") as file:
                self._model = pickle.load(file)
        else:
            self._model = _train_and_save(self._mm)

    def _embed_safe(self, smiles_list: list[str]) -> list:
        try:
            embs = self._mm(smiles_list)
            valid = [emb for emb in embs if isinstance(emb, self._torch.Tensor)]
            if len(valid) == len(smiles_list):
                return embs
        except Exception:
            pass

        results = []
        for smiles in smiles_list:
            try:
                emb = self._mm([smiles])
                results.append(emb[0] if emb and isinstance(emb[0], self._torch.Tensor) else None)
            except Exception:
                results.append(None)
        return results

    def predict(self, smiles: Union[str, list[str]]) -> Union[float, list[float]]:
        import math

        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)
        embs = self._embed_safe(smiles_list)

        valid_idx, valid_embs = [], []
        for idx, emb in enumerate(embs):
            if emb is not None:
                valid_idx.append(idx)
                valid_embs.append(emb)

        preds = [float("nan")] * len(smiles_list)
        if valid_embs:
            valid_preds = self._model.predict(self._torch.stack(valid_embs).numpy()).tolist()
            for idx, pred in zip(valid_idx, valid_preds):
                preds[idx] = pred

        for idx, pred in enumerate(preds):
            if math.isnan(pred):
                preds[idx] = esol(smiles_list[idx])

        return preds[0] if single else preds


__all__ = ["MiniMolSolubility", "esol"]
