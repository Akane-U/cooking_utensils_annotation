import base64
import json
import uuid
from copy import deepcopy
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"

# ─── GitHub storage ───────────────────────────────────────────────────────────
def _gh_headers() -> dict:
    token = st.secrets["github"]["token"]
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def _gh_url(filename: str) -> str:
    cfg = st.secrets["github"]
    return f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}/contents/outputs/{filename}"


def github_read_json(filename: str):
    """outputs/{filename} を GitHub から読み込む。存在しなければ None を返す。"""
    r = requests.get(_gh_url(filename), headers=_gh_headers())
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content = json.loads(base64.b64decode(data["content"]).decode("utf-8"))
    return content, data["sha"]


def github_write_json(filename: str, data) -> None:
    """outputs/{filename} を GitHub に保存（なければ作成、あれば更新）。"""
    cfg = st.secrets["github"]
    _, sha = github_read_json(filename)
    body = {
        "message": f"update {filename}",
        "content": base64.b64encode(
            json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8")
        ).decode("utf-8"),
        "branch": cfg["branch"],
    }
    if sha:
        body["sha"] = sha
    r = requests.put(_gh_url(filename), headers=_gh_headers(), json=body)
    r.raise_for_status()

# ─── Constants ────────────────────────────────────────────────────────────────
# アノテーターID → ログイン名の対応表
ANNOTATORS = {
    "main": "admain",
    "ad":   "adsub1",
    "ad2":  "adsub2",
    "A":    "ayabe",
    "B":    "shibata",
    "C":    "kondo",
}
_NAME_TO_ID = {v: k for k, v in ANNOTATORS.items()}

# sub1/sub2 レシピファイル
SUB_RECIPE_FILES = {
    "sub1": "sub1_recipe_10.json",
    "sub2": "sub2_recipe_10.json",
}
# バッチ固定のアノテーター（ad: sub1専任, ad2: sub2専任）
_FIXED_BATCH = {"ad": "sub1", "ad2": "sub2"}
# ログイン後にsub1/sub2を選択するアノテーター（同じログイン名で両方担当）
_SELECTABLE_BATCH_IDS = {"A", "B", "C"}

UTENSIL_CATEGORIES = {
    "容器・保管可能な器具": (100, 199),
    "加熱容器": (200, 299),
    "切る": (300, 399),
    "混ぜる": (400, 499),
    "すくう": (500, 599),
    "すりおろす・漉す・ふるう": (600, 699),
    "伸ばす・塗る": (700, 799),
    "整える": (800, 899),
    "量る・測る": (900, 999),
    "包む・覆う・敷く": (1000, 1099),
    "道具不使用": (1100, 1199),
}

# vessel（容器・場）扱いのカテゴリ。それ以外は tools（操作道具）扱い。
VESSEL_CATEGORY_NAMES = {"容器・保管可能な器具", "加熱容器"}


def split_utensil_cats(utensil_cats: dict) -> tuple[dict, dict]:
    """utensil_cats を vessel用カテゴリと tools用カテゴリに分割する。"""
    vessel_cats = {k: v for k, v in utensil_cats.items() if k in VESSEL_CATEGORY_NAMES}
    tool_cats = {k: v for k, v in utensil_cats.items() if k not in VESSEL_CATEGORY_NAMES}
    return vessel_cats, tool_cats

# ─── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data
def load_recipes() -> list:
    with open(DATA / "recipe_100.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sub_recipes(batch: str) -> list:
    with open(DATA / SUB_RECIPE_FILES[batch], encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_utensils() -> dict:
    """Returns {category_name: [utensil_name, ...]}"""
    df = pd.read_csv(DATA / "utensils.csv")
    result = {}
    for cat, (lo, hi) in UTENSIL_CATEGORIES.items():
        names = df[df["id"].between(lo, hi)]["name"].tolist()
        if names:
            result[cat] = names
    return result


def flat_utensils(utensil_cats: dict) -> list:
    return [name for names in utensil_cats.values() for name in names]


def build_from_recipes(recipes: list) -> list:
    """レシピJSONからアノテーション初期構造を生成する（resulting_from なし）。"""
    result = []
    for recipe in recipes:
        step0 = [
            {
                "id": str(i),
                "name": ing,
                "utensil_interactions_list": [],
            }
            for i, ing in enumerate(recipe["ingredients"], 1)
        ]
        wsl = [{"step_after": 0, "state_list": step0}]
        for n in range(1, len(recipe["instructions"]) + 1):
            wsl.append({"step_after": n, "state_list": []})
        result.append({"title": recipe["title"], "world_state_list": wsl})
    return result


def clean_for_save(annotations: list) -> list:
    """内部ヘルパーフィールドと resulting_from を除去してからディスクに書き出す。"""
    result = deepcopy(annotations)
    for recipe in result:
        for ws in recipe["world_state_list"]:
            for state in ws["state_list"]:
                state.pop("resulting_from", None)
                for inter in state.get("utensil_interactions_list", []):
                    inter.pop("_uid", None)
    return result


def strip_none_prefixes(annotations: list) -> list:
    """JSON読み込み時に None_ プレフィックスを除去してUI用に正規化する。"""
    result = deepcopy(annotations)
    for recipe in result:
        for ws in recipe.get("world_state_list", []):
            for state in ws.get("state_list", []):
                fp = state.get("final_position", "")
                if fp.startswith("None_"):
                    state["final_position"] = fp[5:]
                for inter in state.get("utensil_interactions_list", []):
                    sid = inter.get("source_state_id", "")
                    if sid.startswith("None_"):
                        inter["source_state_id"] = sid[5:]
                    inter["vessel"] = [
                        v[5:] if v.startswith("None_") else v
                        for v in inter.get("vessel", [])
                    ]
                    inter["tools"] = [
                        t[5:] if t.startswith("None_") else t
                        for t in inter.get("tools", [])
                    ]
    return result


def add_none_prefixes(annotations: list, utensil_cats: dict) -> list:
    """保存前に一覧外の値へ None_ プレフィックスを付与し、final_position を自動導出する。"""
    flat = flat_utensils(utensil_cats)
    result = deepcopy(annotations)
    for recipe in result:
        valid_ids = {
            s["id"]
            for ws in recipe.get("world_state_list", [])
            for s in ws.get("state_list", [])
        }
        for ws in recipe.get("world_state_list", []):
            for state in ws.get("state_list", []):
                for inter in state.get("utensil_interactions_list", []):
                    sid = inter.get("source_state_id", "")
                    if sid and sid not in valid_ids:
                        inter["source_state_id"] = f"None_{sid}"
                    inter["vessel"] = [
                        v if v in flat else f"None_{v}"
                        for v in inter.get("vessel", [])
                    ]
                    inter["tools"] = [
                        t if t in flat else f"None_{t}"
                        for t in inter.get("tools", [])
                    ]
                state["final_position"] = compute_final_position(state)
    return result


def compute_final_position(state: dict) -> str:
    """Stateの最終位置を、合流する最後の生成元(interaction)のvessel配列の末尾から自動導出する。"""
    interactions = state.get("utensil_interactions_list", [])
    if not interactions:
        return ""
    vessel = interactions[-1].get("vessel", [])
    return vessel[-1] if vessel else ""


# ─── State helpers ────────────────────────────────────────────────────────────


def ensure_uids(annotations: list) -> None:
    for recipe in annotations:
        for ws in recipe["world_state_list"]:
            for state in ws["state_list"]:
                for inter in state.get("utensil_interactions_list", []):
                    if "_uid" not in inter:
                        inter["_uid"] = uuid.uuid4().hex[:8]


def get_step_ws(ridx: int, sidx: int):
    for ws in st.session_state.ann[ridx]["world_state_list"]:
        if ws["step_after"] == sidx:
            return ws
    return None


def max_step(ridx: int) -> int:
    return max(ws["step_after"] for ws in st.session_state.ann[ridx]["world_state_list"])


def prev_states(ridx: int, sidx: int) -> dict:
    """Return {id: (step_after, name, final_position)} for all states in steps 0..sidx-1."""
    result = {}
    for ws in st.session_state.ann[ridx]["world_state_list"]:
        if ws["step_after"] < sidx:
            for s in ws["state_list"]:
                result[s["id"]] = (ws["step_after"], s["name"], compute_final_position(s))
    return result


def used_source_ids(ridx: int, sidx: int) -> set:
    """Return all source_state_ids referenced in steps 0..sidx (current step含む)."""
    result = set()
    for ws in st.session_state.ann[ridx]["world_state_list"]:
        if ws["step_after"] <= sidx:
            for state in ws["state_list"]:
                for inter in state.get("utensil_interactions_list", []):
                    sid = inter.get("source_state_id")
                    if sid:
                        result.add(sid)
    return result


def used_utensils_in_recipe(ridx: int) -> set:
    """Return all utensil names used across all steps of the recipe."""
    result = set()
    for ws in st.session_state.ann[ridx]["world_state_list"]:
        for state in ws["state_list"]:
            for inter in state.get("utensil_interactions_list", []):
                for u in inter.get("vessel", []) + inter.get("tools", []):
                    if u:
                        result.add(u)
    return result


def unannotated_indices(ann: list) -> list[int]:
    """全stepを通じてnameが空のstateが1件以上あるレシピのインデックスを返す。"""
    result = []
    for i, recipe in enumerate(ann):
        steps = [ws for ws in recipe["world_state_list"] if ws["step_after"] >= 1]
        has_empty = any(
            not ws["state_list"] or any(not s.get("name", "") for s in ws["state_list"])
            for ws in steps
        )
        if has_empty:
            result.append(i)
    return result


# ─── Session state init ────────────────────────────────────────────────────────


def get_batch(annotator: str) -> str | None:
    """アノテーターが担当するバッチ（'sub1'/'sub2'）。main はフルレシピのため None。"""
    if annotator in _FIXED_BATCH:
        return _FIXED_BATCH[annotator]
    if annotator in _SELECTABLE_BATCH_IDS:
        return st.session_state.get("batch_select")
    return None


def get_recipes(annotator: str) -> list:
    batch = get_batch(annotator)
    return load_sub_recipes(batch) if batch else load_recipes()


def init() -> None:
    annotator = st.session_state.get("annotator_select", "")
    batch = get_batch(annotator)
    key = (annotator, batch)
    prev = st.session_state.get("_ann_key", "__UNSET__")

    if "ann" not in st.session_state or prev != key:
        recipes = load_sub_recipes(batch) if batch else load_recipes()
        ann = build_from_recipes(recipes)
        ensure_uids(ann)
        st.session_state.ann = ann
        st.session_state.ridx = 0
        st.session_state.sidx = 1
        st.session_state._ann_key = key
        if batch:
            fname = f"{annotator}_{batch}_annotated.json"
        else:
            fname = f"{annotator}_annotated.json"
        st.session_state.save_filename = fname
        st.session_state["save_filename_input"] = fname


# ─── Widget helpers ────────────────────────────────────────────────────────────

OTHER = "一覧外（自由記述）"
OTHER_CUSTOM = "一覧外（自由記述）"
_CAT_SEP_PRE = "── "
_CAT_SEP_SUF = " ──"


def utensil_multi_select(label: str, key: str, current: list, utensil_cats: dict) -> list:
    utensils = flat_utensils(utensil_cats)
    known = [u for u in current if u in utensils]
    custom = [u for u in current if u not in utensils]

    opts = []
    for cat, names in utensil_cats.items():
        opts.append(f"{_CAT_SEP_PRE}{cat}{_CAT_SEP_SUF}")
        opts.extend(names)
    opts.append(OTHER)

    default = known + ([OTHER] if custom else [])

    def _remove_seps() -> None:
        st.session_state[key] = [
            u for u in st.session_state[key]
            if not (u.startswith(_CAT_SEP_PRE) and u.endswith(_CAT_SEP_SUF))
        ]

    sel = st.multiselect(label, opts, default=[d for d in default if d in opts], key=key, on_change=_remove_seps)

    result = [
        u for u in sel
        if u != OTHER and not (u.startswith(_CAT_SEP_PRE) and u.endswith(_CAT_SEP_SUF))
    ]
    if OTHER in sel:
        ctext = st.text_input(
            f"{label}（一覧外・カンマ区切り）",
            value=", ".join(custom),
            key=f"{key}_c",
        )
        result += [u.strip() for u in ctext.split(",") if u.strip()]
    return result


def source_label(step: int, name: str) -> str:
    """UIに表示するソースラベル。材料は名前のみ、中間stateは step N: name。"""
    return name if step == 0 else f"step {step}: {name}"


def source_select(label: str, key: str, current: str, src: dict, used_ids: set = None) -> str:
    """src: {id: (step_after, name, final_position)}; used_ids: 既に使用済みのid集合"""
    id2label = {sid: source_label(step, name) for sid, (step, name, _pos) in src.items()}
    label2id = {v: k for k, v in id2label.items()}

    used_labels = {id2label[sid] for sid in (used_ids or []) if sid in id2label}

    cur_label = id2label.get(current, "")
    opts = [""] + list(id2label.values())
    idx = opts.index(cur_label) if cur_label in opts else 0

    def _fmt(v: str) -> str:
        if v in used_labels:
            return "✔ " + v
        return v

    sel = st.selectbox(label, opts, index=idx, key=key, format_func=_fmt)
    return label2id.get(sel, "")


def source_transition_hint(src: dict, source_id: str, current_vessel: list = None) -> str:
    """生成元が中間stateの場合に、使用容器欄に表示する移動道具の案内文を返す（該当なしは""）。"""
    if not source_id or source_id not in src:
        return ""
    step, _name, position = src[source_id]
    if step <= 0:
        return ""
    container_desc = f"「{position}」" if position else "最後に選んだ容器"
    first_desc = f"「{current_vessel[0]}」" if current_vessel else "先頭容器"
    return f"step{step}の{container_desc}⇒{first_desc} に必要な移動道具を先頭に記入"


# ─── Callbacks ────────────────────────────────────────────────────────────────


def cb_add_interaction(ridx, sidx, si):
    ws = get_step_ws(ridx, sidx)
    ws["state_list"][si]["utensil_interactions_list"].append(
        {"source_state_id": "", "vessel": [], "tools": [], "_uid": uuid.uuid4().hex[:8]}
    )


def cb_del_interaction(ridx, sidx, si, ii):
    ws = get_step_ws(ridx, sidx)
    ws["state_list"][si]["utensil_interactions_list"].pop(ii)


def cb_add_state(ridx, sidx):
    ws = get_step_ws(ridx, sidx)
    ws["state_list"].append(
        {
            "id": uuid.uuid4().hex[:8],
            "name": "",
            "utensil_interactions_list": [],
        }
    )


def cb_del_state(ridx, sidx, si):
    ws = get_step_ws(ridx, sidx)
    ws["state_list"].pop(si)


# ─── Main ─────────────────────────────────────────────────────────────────────


def _login_screen() -> None:
    st.title("調理器具アノテーション")
    st.markdown("#### あなたの名字を半角ローマ字で入力して開始してください")
    entered = st.text_input("名前")

    if st.button("開始", type="primary"):
        if entered not in _NAME_TO_ID:
            st.error(f"名前が正しくありません: {entered}")
            return
        st.session_state.annotator_select = _NAME_TO_ID[entered]
        st.session_state.annotator_confirmed = True
        st.rerun()


def _batch_screen() -> None:
    st.title("調理器具アノテーション")
    annotator = st.session_state.get("annotator_select", "")
    label = ANNOTATORS.get(annotator, "")
    st.markdown(f"#### {label} さん、担当するレシピセットを選択してください")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("sub1（10レシピ）", type="primary", use_container_width=True):
            st.session_state.batch_select = "sub1"
            st.rerun()
    with col2:
        if st.button("sub2（10レシピ）", type="primary", use_container_width=True):
            st.session_state.batch_select = "sub2"
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="アノテーションツール", layout="wide")

    if not st.session_state.get("annotator_confirmed", False):
        _login_screen()
        return

    annotator = st.session_state.get("annotator_select", "")
    if annotator in _SELECTABLE_BATCH_IDS and not st.session_state.get("batch_select"):
        _batch_screen()
        return

    init()

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
        /* 全カラム独立スクロール */
        section[data-testid="stMain"]
            div[data-testid="stHorizontalBlock"]
            > div[data-testid="stColumn"] {
            position: sticky;
            top: 0;
            max-height: 100vh;
            overflow-y: auto;
        }
        /* CUD: state カード — 青(#005AFF)左アクセント + 薄青背景 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #EFF7FF !important;
            border-left: 4px solid #005AFF !important;
        }
        /* CUD: primary ボタン（stepナビ・保存）をオレンジ→緑 */
        button[data-testid="stBaseButton-primary"] {
            background-color: #03AF7A !important;
            border-color: #03AF7A !important;
            color: #fff !important;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #029468 !important;
            border-color: #029468 !important;
        }
        /* CUD: multiselect 選択チップをオレンジ→緑 */
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background-color: #03AF7A !important;
        }
        /* CUD: multiselect の "Select all" を非表示 */
        div[data-testid="stMultiSelect"] ul li:first-child:has(input[type="checkbox"]) {
            display: none !important;
        }
        /* 備考テキストエリアを小さめに */
        textarea[data-testid="stTextArea"] {
            font-size: 0.82em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    utensil_cats = load_utensils()
    vessel_cats, tool_cats = split_utensil_cats(utensil_cats)
    annotator = st.session_state.get("annotator_select", "")
    batch = get_batch(annotator)
    recipes = get_recipes(annotator)
    ann = st.session_state.ann

    # ── 4カラム: ナビ | レシピ情報 | アノテーション | 器具一覧 ───────────────────────
    nav_col, left, mid, utensil_col = st.columns([1, 2, 5, 1], gap="large")

    # ── Nav column ────────────────────────────────────────────────────────────
    with nav_col:
        # st.markdown("**アノテーション**")

        annotator_label = ANNOTATORS.get(annotator, "admin")
        label_suffix = f"（{batch}）" if batch else ""
        st.markdown(f"**{annotator_label}{label_suffix}**")

        if st.button("ログアウト", use_container_width=True):
            for k in ["annotator_confirmed", "annotator_select", "batch_select", "_ann_key", "ann"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()

        new_ridx = st.selectbox(
            "レシピ選択",
            range(len(recipes)),
            format_func=lambda i: f"{i + 1}. {recipes[i]['title']}",
            index=st.session_state.ridx,
            key="sb_recipe",
        )
        if new_ridx != st.session_state.ridx:
            st.session_state.ridx = new_ridx
            st.session_state.sidx = 1
            st.rerun()

        ridx = st.session_state.ridx
        mstep = max_step(ridx)

        st.divider()

        for si in range(1, mstep + 1):
            btype = "primary" if si == st.session_state.sidx else "secondary"
            if st.button(f"Step {si}", key=f"nav_{si}", type=btype, use_container_width=True):
                st.session_state.sidx = si
                st.rerun()

        st.divider()

        filename = st.session_state.save_filename
        if not filename.endswith(".json"):
            filename += ".json"

        if st.button("☁ 保存", type="primary", use_container_width=True):
            try:
                github_write_json(
                    filename,
                    add_none_prefixes(clean_for_save(ann), utensil_cats),
                )
                st.success(f"保存しました: outputs/{filename}")
            except Exception as e:
                st.error(f"保存失敗: {e}")

        if st.button("☁ 読み込み", use_container_width=True):
            try:
                loaded, _ = github_read_json(filename)
                if loaded is None:
                    st.warning(f"outputs/{filename} がストレージに見つかりません")
                else:
                    fresh = build_from_recipes(recipes)
                    loaded_stripped = strip_none_prefixes(loaded)
                    loaded_map = {r["title"]: r for r in loaded_stripped}
                    for i, r in enumerate(fresh):
                        if r["title"] in loaded_map:
                            fresh[i] = loaded_map[r["title"]]
                    ensure_uids(fresh)
                    st.session_state.ann = fresh
                    st.session_state.ridx = 0
                    st.session_state.sidx = 1
                    st.rerun()
            except Exception as e:
                st.error(f"読み込み失敗: {e}")

        unannotated = unannotated_indices(ann)
        if unannotated:
            st.divider()
            st.markdown(f"**未アノテーション：{len(unannotated)}件**")
            st.caption(f"「{recipes[unannotated[0]]['title']}」から再開")

    ridx = st.session_state.ridx
    sidx = st.session_state.sidx
    recipe = recipes[ridx]

    # ── Left column: recipe info ───────────────────────────────────────────────
    with left:
        st.subheader(recipe["title"])

        with st.expander("材料", expanded=False):
            for ing in recipe["ingredients"]:
                st.write(f"• {ing}")

        st.markdown("#### 調理手順")
        for i, instr in enumerate(recipe["instructions"], 1):
            if i == sidx:
                st.markdown(
                    f'<div style="background:#fff9c4;padding:10px;border-radius:6px;'
                    f'border-left:4px solid #f9a825;margin:4px 0">'
                    f"<b>Step </b> {instr}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**Step** {instr}")
            st.write("")

        st.markdown("#### アノテーション備考欄")
        ann[ridx]["annotation_note"] = st.text_area(
            "アノテーション備考欄",
            value=ann[ridx].get("annotation_note", ""),
            key=f"annotation_note_{ridx}",
            height=120,
            placeholder="不明点・迷った点・感じた点・改善すべき点（Step1: ○○が不明など）（複数ある場合は改行して区切る）",
            label_visibility="collapsed",
        )

    # ── Utensil column ────────────────────────────────────────────────────────
    with utensil_col:
        st.markdown("**🥄 器具一覧**")
        st.divider()
        used = used_utensils_in_recipe(ridx)
        mark_cats = {"容器・保管可能な器具", "加熱容器"}
        for cat, names in utensil_cats.items():
            with st.expander(cat, expanded=(cat in mark_cats)):
                for u in names:
                    if cat in mark_cats and u in used:
                        st.markdown(
                            f'<span style="color:#03AF7A;font-weight:bold">✔ {u}</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(u)

    # ── Middle column: annotation form ────────────────────────────────────────
    with mid:
        st.markdown(f"#### Step {sidx} アノテーション")

        step_ws = get_step_ws(ridx, sidx)
        if step_ws is None:
            st.error("このステップのデータが見つかりません")
            return

        src = prev_states(ridx, sidx)
        used_sources = used_source_ids(ridx, sidx)

        # stateが空なら1つ自動追加
        if not step_ws["state_list"]:
            step_ws["state_list"].append(
                {
                    "id": uuid.uuid4().hex[:8],
                    "name": "",
                    "utensil_interactions_list": [],
                }
            )

        flat_vessel = flat_utensils(vessel_cats)
        flat_tools = flat_utensils(tool_cats)
        state_to_del = None
        for si, state in enumerate(step_ws["state_list"]):
            with st.container(border=True):
                h_col, del_col = st.columns([8, 1])
                with h_col:
                    st.markdown(
                        f'<span style="background:#005AFF;color:#fff;'
                        f'padding:3px 10px;border-radius:4px;font-size:0.9em;font-weight:bold">'
                        f'State {si + 1}</span>',
                        unsafe_allow_html=True,
                    )
                with del_col:
                    if st.button("🗑", key=f"dst_{ridx}_{sidx}_{si}", help="このStateを削除"):
                        state_to_del = si

                # 最終ステップは名前にレシピタイトルを必ず入れる
                if sidx == mstep and not state.get("name"):
                    state["name"] = recipe["title"]

                state["name"] = st.text_input(
                    "名前（name）",
                    value=state.get("name", ""),
                    key=f"name_{ridx}_{sidx}_{si}",
                )

                st.markdown("---")
                st.markdown("**生成元（材料一覧・登録済 State） → 使用容器（vessel）・使用道具（tools）**")

                interactions = state.setdefault("utensil_interactions_list", [])

                # 生成元が空なら1つ自動追加
                if not interactions:
                    interactions.append(
                        {"source_state_id": "", "vessel": [], "tools": [], "_uid": uuid.uuid4().hex[:8]}
                    )

                to_del = None
                for ii, inter in enumerate(interactions):
                    uid = inter.setdefault("_uid", uuid.uuid4().hex[:8])
                    wkey = f"u_{ridx}_{sidx}_{si}_{uid}"

                    with st.container():
                        src_col, vessel_col, tools_col, copy_col, del_col = st.columns([4, 4, 4, 1, 1])

                        with src_col:
                            inter["source_state_id"] = source_select(
                                "生成元（source_state_id）",
                                f"src_{ridx}_{sidx}_{si}_{uid}",
                                inter.get("source_state_id", ""),
                                src,
                                used_ids=used_sources,
                            )

                        with vessel_col:
                            inter["vessel"] = utensil_multi_select(
                                "使用容器（vessels）※複数選択可",
                                f"{wkey}_vessel",
                                inter.get("vessel", []),
                                vessel_cats,
                            )

                        with tools_col:
                            inter["tools"] = utensil_multi_select(
                                "使用道具（tools）※複数選択可",
                                f"{wkey}_tools",
                                inter.get("tools", []),
                                tool_cats,
                            )
                            src_hint = source_transition_hint(
                                src, inter["source_state_id"], inter["vessel"]
                            )
                            if src_hint:
                                st.warning(src_hint)
                            source_step = src.get(inter["source_state_id"], (0, "", ""))[0]
                            if sidx == mstep and source_step > 0:
                                st.warning("「盛り付け皿・器」へ移動するために必要な移動道具を末尾に記入")

                        with copy_col:
                            if ii > 0:
                                prev_vessel = deepcopy(interactions[ii - 1].get("vessel", []))
                                prev_tools = deepcopy(interactions[ii - 1].get("tools", []))

                                def _do_copy(
                                    _inter=inter,
                                    _prev_vessel=prev_vessel,
                                    _prev_tools=prev_tools,
                                    _wkey=wkey,
                                    _flat_vessel=flat_vessel,
                                    _flat_tools=flat_tools,
                                ) -> None:
                                    for _field, _prev, _flat in (
                                        ("vessel", _prev_vessel, _flat_vessel),
                                        ("tools", _prev_tools, _flat_tools),
                                    ):
                                        current_u = _inter.get(_field, [])
                                        merged = list(dict.fromkeys(current_u + _prev))
                                        _inter[_field] = merged
                                        in_list = [u for u in merged if u in _flat]
                                        custom_p = [u for u in merged if u not in _flat]
                                        _fkey = f"{_wkey}_{_field}"
                                        st.session_state[_fkey] = in_list + (
                                            [OTHER] if custom_p else []
                                        )
                                        if custom_p:
                                            st.session_state[f"{_fkey}_c"] = ", ".join(custom_p)
                                        else:
                                            st.session_state.pop(f"{_fkey}_c", None)

                                st.write("")
                                st.button(
                                    "⬆",
                                    key=f"copy_{ridx}_{sidx}_{si}_{uid}",
                                    help="1個上の行の容器・道具をコピー",
                                    on_click=_do_copy,
                                )

                        with del_col:
                            st.write("")
                            if st.button("🗑", key=f"del_{ridx}_{sidx}_{si}_{uid}"):
                                to_del = ii

                if to_del is not None:
                    cb_del_interaction(ridx, sidx, si, to_del)
                    st.rerun()

                if st.button("＋ ソースを追加", key=f"add_{ridx}_{sidx}_{si}"):
                    cb_add_interaction(ridx, sidx, si)
                    st.rerun()

        if state_to_del is not None:
            cb_del_state(ridx, sidx, state_to_del)
            st.rerun()

        st.markdown("---")
        if st.button("＋ Stateを追加", key=f"addst_{ridx}_{sidx}"):
            cb_add_state(ridx, sidx)
            st.rerun()


if __name__ == "__main__":
    main()
