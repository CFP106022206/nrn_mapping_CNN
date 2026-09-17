"""把 VFB 的 FlyCircuit 型別對上 neuPrint 的 hemibrain 型別，並判定一組配對是否可能同型。

判定分四級，仿照專家信心度的寫法：

    1.0  型別相符          -> 應該是同一型神經（可當正例）
    0.5  同亞族／僅能確認到 lineage 層級 -> 形態近親，無法斷定
    0.0  型別確定不同      -> 一定不是同一顆（可當負例）
    (空) EM 端未定型        -> 無法判斷，不納入統計

「亞族」的粒度刻意跟著參考資料的粒度走：

- LC 神經：VFB 的 `LC12` 已是最細單位，LC12 vs LC10 就是確定不同 -> 0.0
- Kenyon cell：VFB 只分 core/surface/posterior，hemibrain 另有 KCab-m 等切法，
  同屬 alpha/beta 的互配給 0.5 而不是 0.0
- 嗅覺投射神經：以腎小球（glomerulus）為亞族，DL2d_adPN vs DL2d_vPN 給 0.5，
  DL2d vs DL2v 給 0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- EM 端分類

_VISUAL = re.compile(
    r"^(LC\d|LPLC|LLPC|LPC|LT\d|LTe|MC\d|LO[^a-z]|LPT|Tm|TmY|T[1-5]|Mi\d|Dm\d|Pm\d|Y\d|Li\d)"
)
_CX = re.compile(
    r"^(EPG|PEG|PEN|PFN|PFL|PFR|FS\d|FC\d|FB\d|FR\d|ER\d|EL\b|hDelta|vDelta|ExR|SpsP|IbSpsP|P6-8P9)"
)
# 同一腎小球組合有多型時 hemibrain 會在字尾加編號（`VP1d+VP4_l2PN1`），也要算 PN
_PN_SUFFIX = re.compile(r"_(adPN|lPN|vPN|ilPN|l2PN|lvPN|adPNm\d?|m?lPN)\d*$")

# hemibrain 的 PN 命名有兩種形態：
#   單腎小球  `VM5d_adPN`      -> 譜系 token 在字尾
#   多腎小球  `M_l2PNl20`      -> `M_<譜系><編號>`，譜系 token 在中間
# 兩者都要抓得到，所以用 search 而不是 endswith。
_PN_LINEAGE = re.compile(r"(?:^M_|_)(ad|il|lv|l|v|sp|b)\d*PN")

# VFB 的譜系用語 -> hemibrain 的譜系 token
_LINEAGE_TOKEN = {
    "ALl1": "l",      # 側向譜系，涵蓋 lPN 與 l2PN
    "adPN": "ad",
    "vPN": "v",
    "lPN": "l",
}


def pn_lineage(np_type: str) -> str:
    """回傳 hemibrain PN 型別的譜系 token（ad / l / v / lv / il / sp），抓不到回空字串。"""
    if not np_type:
        return ""
    m = _PN_LINEAGE.search(np_type)
    return m.group(1) if m else ""


def em_family(np_type: str) -> str:
    if not np_type:
        return ""
    if np_type.startswith("KC"):
        return "KC"
    if _PN_SUFFIX.search(np_type) or np_type.startswith("M_"):
        return "ALPN"
    if _VISUAL.match(np_type):
        return "visual"
    if _CX.match(np_type):
        return "CX"
    return "other"


def em_subfamily(np_type: str) -> str:
    """回傳與參考資料同粒度的亞族代碼。"""
    if not np_type:
        return ""
    if np_type.startswith("KC"):
        m = re.match(r"^(KCab|KCg|KCa'b')", np_type)
        return m.group(1) if m else "KC"
    m = _PN_SUFFIX.search(np_type)
    if m:
        return np_type[: m.start()]  # 腎小球，如 DL2d
    return np_type  # 視覺等族群，型別本身就是最細單位


# ---------------------------------------------------------------- FC 端期望

@dataclass
class Expectation:
    """一個 VFB 型別標註所對應的 hemibrain 期望值。"""

    family: str
    exact: set[str] = field(default_factory=set)
    prefix: str = ""          # 前綴即可視為相符（VFB 標到亞型層級）
    subfamily: str = ""       # 用於 same_subfamily 判定
    lineage: str = ""         # PN 譜系 token，如 'l'（ALl1）、'ad'、'v'
    vague: bool = False       # VFB 標註粒度粗於 neuPrint
    label: str = ""           # 原始 VFB 字串


_KC_EXACT = {
    "alpha/beta core Kenyon cell": ("KCab-c", "KCab"),
    "alpha/beta surface Kenyon cell": ("KCab-s", "KCab"),
    "alpha/beta posterior Kenyon cell": ("KCab-p", "KCab"),
    "gamma dorsal Kenyon cell": ("KCg-d", "KCg"),
}
_KC_PREFIX = {
    "adult gamma Kenyon cell": ("KCg-", "KCg"),
    "adult alpha'/beta' Kenyon cell": ("KCa'b'-", "KCa'b'"),
}

_LC_RE = re.compile(r"^lobula columnar neuron (LC\d+)$")
_PN_FULL_RE = re.compile(r"^adult antennal lobe projection neuron (\S+) (adPN|vPN|lPN)$")
_PN_GLOM_RE = re.compile(r"^adult antennal lobe projection neuron (\S+)$")


def expectation(vfb_type: str) -> Expectation | None:
    """把一個 VFB 型別字串翻成 hemibrain 期望值；無法對應則回 None。"""
    t = vfb_type.strip()

    m = _LC_RE.match(t)
    if m:
        lc = m.group(1)
        return Expectation(family="visual", exact={lc}, subfamily=lc, label=t)

    if t in _KC_EXACT:
        exact, sub = _KC_EXACT[t]
        return Expectation(family="KC", exact={exact}, subfamily=sub, label=t)
    if t in _KC_PREFIX:
        pre, sub = _KC_PREFIX[t]
        return Expectation(family="KC", prefix=pre, subfamily=sub, label=t)

    m = _PN_FULL_RE.match(t)
    if m:
        glom, lineage = m.group(1), m.group(2)
        return Expectation(
            family="ALPN", exact={f"{glom}_{lineage}"}, subfamily=glom, label=t
        )

    if t == "antennal lobe projection neuron of ALl1 lineage":
        # ALl1 = 側向譜系，hemibrain 寫成 lPN 或 l2PN，都算相容
        return Expectation(
            family="ALPN", lineage=_LINEAGE_TOKEN["ALl1"], vague=True, label=t
        )
    if t.startswith("adult multiglomerular antennal lobe projection neuron"):
        tail = t.rsplit(" ", 1)[-1]
        return Expectation(
            family="ALPN", prefix="M_", lineage=_LINEAGE_TOKEN.get(tail, ""),
            vague=True, label=t,
        )
    if t.startswith("adult uniglomerular antennal lobe projection neuron"):
        tail = t.rsplit(" ", 1)[-1]
        return Expectation(
            family="ALPN", lineage=_LINEAGE_TOKEN.get(tail, ""), vague=True, label=t
        )

    m = _PN_GLOM_RE.match(t)
    if m:
        glom = m.group(1)
        return Expectation(
            family="ALPN", prefix=f"{glom}_", subfamily=glom, label=t
        )

    return None


# ---------------------------------------------------------------- 判定

def grade(exp: Expectation, np_type: str, np_instance: str = "") -> tuple[str, float | None, str]:
    """回傳 (relation, label, basis)。"""
    if not np_type:
        inst = f"，instance={np_instance}" if np_instance else ""
        return "em_untyped", None, f"hemibrain 未給正式 type{inst}，無法判斷"

    fam, sub = em_family(np_type), em_subfamily(np_type)

    if exp.exact and np_type in exp.exact:
        return "exact", 1.0, f"VFB「{exp.label}」= hemibrain {np_type}，型別相符"
    if exp.prefix and np_type.startswith(exp.prefix) and not exp.vague:
        return "exact", 1.0, f"VFB「{exp.label}」涵蓋 hemibrain {np_type}，型別相符"

    if exp.vague:
        # 至少要同大類。「adult uniglomerular antennal lobe projection neuron」
        # 沒有 prefix 也沒有譜系可檢查，少了這條會把 PFNd、DNp23 也判成相容
        ok = fam == exp.family
        if exp.prefix and not np_type.startswith(exp.prefix):
            ok = False
        if exp.lineage and pn_lineage(np_type) != exp.lineage:
            ok = False
        if ok:
            return (
                "lineage_ok",
                0.5,
                f"VFB 僅標到「{exp.label}」，hemibrain {np_type} 與之相容但無法確認到型別",
            )

    if exp.subfamily and sub == exp.subfamily:
        return (
            "same_subfamily",
            0.5,
            f"同亞族（{sub}）但亞型不同：VFB「{exp.label}」vs hemibrain {np_type}",
        )

    if fam == exp.family:
        return (
            "same_family",
            0.0,
            f"同大類（{fam}）但型別確定不同：VFB「{exp.label}」vs hemibrain {np_type}",
        )

    return (
        "different_family",
        0.0,
        f"完全不同類群：VFB「{exp.label}」({exp.family}) vs hemibrain {np_type}"
        f"{'(' + fam + ')' if fam else ''}",
    )
