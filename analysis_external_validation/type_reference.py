"""外部型別參考表的抓取與快取。

兩個來源都獨立於本專案的 CNN：

- FC 端：Virtual Fly Brain 的 FlyCircuit 神經策展型別。VFB 把每顆 FlyCircuit 神經
  以 INSTANCEOF 連到 FBbt 本體論的細胞型別（文獻定義，例如 LC12 出自 Wu et al. 2016）。
- EM 端：neuPrint hemibrain v1.2.1 的 `type` / `instance` 欄位（FlyEM 策展）。

兩邊都用公開端點，不需要 token。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

VFB_CYPHER_URL = "https://pdb.virtualflybrain.org/db/neo4j/tx/commit"
VFB_AUTH = "neo4j:neo4j"
NEUPRINT_URL = "https://neuprint.janelia.org/api/custom/custom"
NEUPRINT_DATASET = "hemibrain:v1.2.1"

# VFB 裡 FlyCircuit 神經的泛用標註，不帶型別資訊
GENERIC_VFB_LABELS = {
    "adult neuron",
    "expression pattern fragment",
    "neuron",
    "cell",
}


def _post(url: str, payload: dict, auth: str | None = None, timeout: int = 300) -> dict:
    cmd = ["curl", "-s", "-H", "Content-Type: application/json"]
    if auth:
        cmd += ["-u", auth]
    cmd += ["--data-binary", json.dumps(payload), url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    return json.loads(out)


def vfb_query(cypher: str) -> pd.DataFrame:
    d = _post(VFB_CYPHER_URL, {"statements": [{"statement": cypher}]}, auth=VFB_AUTH)
    if d.get("errors"):
        raise RuntimeError(d["errors"])
    res = d["results"][0]
    return pd.DataFrame([r["row"] for r in res["data"]], columns=res["columns"])


def neuprint_query(cypher: str) -> pd.DataFrame:
    d = _post(NEUPRINT_URL, {"dataset": NEUPRINT_DATASET, "cypher": cypher})
    if "data" not in d:
        raise RuntimeError(d)
    return pd.DataFrame(d["data"], columns=d["columns"])


def fetch_fc_types() -> pd.DataFrame:
    """FlyCircuit 神經 -> VFB 策展型別。一顆神經可能有多個標註，全部保留。"""
    generic = ", ".join(f"'{x}'" for x in sorted(GENERIC_VFB_LABELS))
    df = vfb_query(
        f"""
        MATCH (m:Individual)-[:has_source]->(:DataSet {{short_form:'Chiang2010'}})
        MATCH (m)-[:INSTANCEOF]->(t:Class)
        WHERE NOT t.label IN [{generic}]
        RETURN m.label AS fc_id, t.label AS vfb_type
        """
    )
    return df.drop_duplicates()


def fetch_em_types(body_ids, batch: int = 800) -> pd.DataFrame:
    """hemibrain bodyId -> neuPrint type / instance / status。"""
    ids = [str(int(b)) for b in body_ids]
    frames = []
    for i in range(0, len(ids), batch):
        chunk = ",".join(ids[i : i + batch])
        frames.append(
            neuprint_query(
                f"MATCH (n:Neuron) WHERE n.bodyId IN [{chunk}] "
                "RETURN n.bodyId AS em_id, n.type AS np_type, "
                "n.instance AS np_instance, n.statusLabel AS np_status"
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_or_fetch(path: Path, fetch, **kwargs) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    df = fetch(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
