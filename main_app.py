import base64
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
OUTPUTS = BASE / "outputs"

GROUND_TRUTH_FILE = "main_annotated.json"

# ─── GitHub storage（レビュー結果の保存・読み込み）─────────────────────────────
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
# 本番アノテーション（main_annotated.json）の正誤判定を行うアノテーター
ANNOTATORS = {
    "A": "ayabe",
    "B": "shibata",
    "C": "kondo",
}
_NAME_TO_ID = {v: k for k, v in ANNOTATORS.items()}

UTENSIL_CATEGORIES = {
    "容器・保管可能な器具": (100, 199),
    "加熱容器": (200, 299),
    "切る": (300, 399),
    "混ぜる": (400, 499),
    "すくう": (500, 599),
    "すりおろす・漉す・ふるう": (600, 699),
    "伸ばす・塗る": (700, 799),
    "整える": (800, 899),
    "包む・覆う・敷く": (900, 999),
    "道具不使用": (1000, 1099),
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
def load_ground_truth() -> list:
    """正誤判定の元データ（本番アノテーション結果、正解として扱う）。"""
    with open(OUTPUTS / GROUND_TRUTH_FILE, encoding="utf-8") as f:
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


# ─── Review structure ─────────────────────────────────────────────────────────


def _strip_none(value: str) -> str:
    """一覧外の値に付与された None_ プレフィックスを表示用に除去する。"""
    return value[5:] if value.startswith("None_") else value


def build_review(ground_truth: list) -> list:
    """main_annotated.json から正誤判定用の構造を作る（check/commentフィールドを付与）。
    name / source_state_id / vessel / tools は固定値としてそのまま保持する
    （None_ プレフィックスのみ表示用に除去する）。"""
    result = deepcopy(ground_truth)
    for recipe in result:
        recipe.setdefault("reviewer_note", "")
        recipe.setdefault("reviewed", False)
        for ws in recipe["world_state_list"]:
            for state in ws["state_list"]:
                state["final_position"] = _strip_none(state.get("final_position", ""))
            if ws["step_after"] == 0:
                continue
            for state in ws["state_list"]:
                state.setdefault("name_check", False)
                state.setdefault("name_comment", "")
                for ii, inter in enumerate(state.get("utensil_interactions_list", [])):
                    inter["_uid"] = f"{state['id']}_{ii}"
                    inter["source_state_id"] = _strip_none(inter.get("source_state_id", ""))
                    inter["vessel"] = [_strip_none(v) for v in inter.get("vessel", [])]
                    inter["tools"] = [_strip_none(t) for t in inter.get("tools", [])]
                    inter.setdefault("source_state_id_check", False)
                    inter.setdefault("source_state_id_comment", "")
                    inter.setdefault("vessel_check", False)
                    inter.setdefault("vessel_comment", "")
                    inter.setdefault("tools_check", False)
                    inter.setdefault("tools_comment", "")
    return result


def clean_for_save(review: list) -> list:
    """内部ヘルパーフィールドを除去してからディスクに書き出す。"""
    result = deepcopy(review)
    for recipe in result:
        for ws in recipe["world_state_list"]:
            for state in ws["state_list"]:
                for inter in state.get("utensil_interactions_list", []):
                    inter.pop("_uid", None)
    return result


def merge_saved_review(fresh: list, saved: list) -> list:
    """既存の保存済みレビュー結果（check/comment）を、最新のground truthから
    作った構造にマージする。固定値（name/vessel/tools等）は常にfreshを正とする。"""
    saved_map = {r["title"]: r for r in saved}
    for recipe in fresh:
        sr = saved_map.get(recipe["title"])
        if not sr:
            continue
        recipe["reviewer_note"] = sr.get("reviewer_note", "")
        recipe["reviewed"] = sr.get("reviewed", False)
        saved_states = {
            s["id"]: s
            for ws in sr.get("world_state_list", [])
            for s in ws.get("state_list", [])
        }
        for ws in recipe["world_state_list"]:
            for state in ws["state_list"]:
                ss = saved_states.get(state["id"])
                if not ss:
                    continue
                state["name_check"] = ss.get("name_check", False)
                state["name_comment"] = ss.get("name_comment", "")
                saved_inters = ss.get("utensil_interactions_list", [])
                for ii, inter in enumerate(state.get("utensil_interactions_list", [])):
                    if ii >= len(saved_inters):
                        break
                    si = saved_inters[ii]
                    inter["source_state_id_check"] = si.get("source_state_id_check", False)
                    inter["source_state_id_comment"] = si.get("source_state_id_comment", "")
                    inter["vessel_check"] = si.get("vessel_check", False)
                    inter["vessel_comment"] = si.get("vessel_comment", "")
                    inter["tools_check"] = si.get("tools_check", False)
                    inter["tools_comment"] = si.get("tools_comment", "")
    return fresh


def build_fresh_ann() -> list:
    """ground truth から、recipes と同じ順序の正誤判定用構造を作る。"""
    ground_truth = load_ground_truth()
    fresh = build_review(ground_truth)
    recipes = load_recipes()
    by_title = {r["title"]: r for r in fresh}
    return [by_title[r["title"]] for r in recipes]


def source_label(step: int, name: str) -> str:
    """生成元の表示ラベル。材料は名前のみ、中間stateは step N: name。"""
    return name if step == 0 else f"step {step}: {name}"


def build_id_index(recipe: dict) -> dict:
    """recipe内の全id → (step_after, name, final_position)"""
    result = {}
    for ws in recipe["world_state_list"]:
        for s in ws["state_list"]:
            result[s["id"]] = (ws["step_after"], s.get("name", ""), s.get("final_position", ""))
    return result


# ─── State helpers ────────────────────────────────────────────────────────────


def get_step_ws(ridx: int, sidx: int):
    for ws in st.session_state.ann[ridx]["world_state_list"]:
        if ws["step_after"] == sidx:
            return ws
    return None


def max_step(ridx: int) -> int:
    return max(ws["step_after"] for ws in st.session_state.ann[ridx]["world_state_list"])


def used_vessels_tools_in_recipe(ridx: int) -> set:
    """Return all vessel/tool names used across all steps of the recipe."""
    result = set()
    for ws in st.session_state.ann[ridx]["world_state_list"]:
        for state in ws["state_list"]:
            for inter in state.get("utensil_interactions_list", []):
                for u in inter.get("vessel", []) + inter.get("tools", []):
                    if u:
                        result.add(u)
    return result


def unreviewed_indices(ann: list) -> list[int]:
    """「確認完了」チェックがまだ付いていないレシピのインデックスを返す。"""
    return [i for i, recipe in enumerate(ann) if not recipe.get("reviewed", False)]


# ─── Session state init ────────────────────────────────────────────────────────


def init() -> None:
    annotator = st.session_state.get("annotator_select", "")
    if "ann" not in st.session_state or st.session_state.get("_ann_key") != annotator:
        st.session_state.ann = build_fresh_ann()
        st.session_state.ridx = 0
        st.session_state.sidx = 1
        st.session_state._ann_key = annotator
        st.session_state.save_filename = f"{annotator}_main_reviewed.json"


# ─── Main ─────────────────────────────────────────────────────────────────────


def _login_screen() -> None:
    st.title("調理器具アノテーション 正誤判定")
    st.markdown("#### あなたの名字を半角ローマ字で入力して開始してください")
    entered = st.text_input("名前")

    if st.button("開始", type="primary"):
        if entered not in _NAME_TO_ID:
            st.error(f"名前が正しくありません: {entered}")
            return
        st.session_state.annotator_select = _NAME_TO_ID[entered]
        st.session_state.annotator_confirmed = True
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="正誤判定ツール", layout="wide")

    if not st.session_state.get("annotator_confirmed", False):
        _login_screen()
        return

    annotator = st.session_state.get("annotator_select", "")
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
        /* 備考・コメントテキストエリアを小さめに */
        textarea[data-testid="stTextArea"] {
            font-size: 0.82em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    utensil_cats = load_utensils()
    recipes = load_recipes()
    ann = st.session_state.ann

    # ── 4カラム: ナビ | レシピ情報 | 正誤判定 | 器具一覧 ───────────────────────
    nav_col, left, mid, utensil_col = st.columns([1, 2, 5, 1], gap="large")

    # ── Nav column ────────────────────────────────────────────────────────────
    with nav_col:
        annotator_label = ANNOTATORS.get(annotator, "")
        st.markdown(f"**{annotator_label}**")

        if st.button("ログアウト", use_container_width=True):
            for k in ["annotator_confirmed", "annotator_select", "_ann_key", "ann"]:
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

        if st.button("☁ 保存", type="primary", use_container_width=True):
            try:
                github_write_json(filename, clean_for_save(ann))
                st.success(f"保存しました: outputs/{filename}")
            except Exception as e:
                st.error(f"保存失敗: {e}")

        if st.button("☁ 読み込み", use_container_width=True):
            try:
                loaded, _ = github_read_json(filename)
                if loaded is None:
                    st.warning(f"outputs/{filename} がストレージに見つかりません")
                else:
                    fresh = build_fresh_ann()
                    st.session_state.ann = merge_saved_review(fresh, loaded)
                    st.session_state.ridx = 0
                    st.session_state.sidx = 1
                    st.rerun()
            except Exception as e:
                st.error(f"読み込み失敗: {e}")

        unreviewed = unreviewed_indices(ann)
        if unreviewed:
            st.divider()
            st.markdown(f"**未確認：{len(unreviewed)}件**")
            st.caption(f"「{recipes[unreviewed[0]]['title']}」から再開")

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

        admin_note = ann[ridx].get("annotation_note", "")
        if admin_note:
            with st.expander("作成者の備考", expanded=False):
                st.write(admin_note)

        st.markdown("#### レビューコメント（レシピ全体）")
        ann[ridx]["reviewer_note"] = st.text_area(
            "レビューコメント",
            value=ann[ridx].get("reviewer_note", ""),
            key=f"reviewer_note_{ridx}",
            height=100,
            placeholder="レシピ全体を通しての気づき・コメント（複数ある場合は改行して区切る）",
            label_visibility="collapsed",
        )
        ann[ridx]["reviewed"] = st.checkbox(
            "✅ このレシピの確認完了",
            value=ann[ridx].get("reviewed", False),
            key=f"reviewed_{ridx}",
        )

    # ── Utensil column ────────────────────────────────────────────────────────
    with utensil_col:
        st.markdown("**🥄 器具一覧（参考）**")
        st.divider()
        used = used_vessels_tools_in_recipe(ridx)
        mark_cats = {"容器・保管可能な器具", "加熱容器"}
        for cat, names in utensil_cats.items():
            with st.expander(cat, expanded=(cat in mark_cats)):
                for u in names:
                    if u in used:
                        st.markdown(
                            f'<span style="color:#03AF7A;font-weight:bold">✔ {u}</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(u)

    # ── Middle column: 正誤判定フォーム ────────────────────────────────────────
    with mid:
        st.markdown(f"#### Step {sidx} 正誤判定")

        step_ws = get_step_ws(ridx, sidx)
        if step_ws is None or not step_ws["state_list"]:
            st.info("このステップに生成物はありません")
        else:
            id_index = build_id_index(ann[ridx])

            for si, state in enumerate(step_ws["state_list"]):
                with st.container(border=True):
                    st.markdown(
                        f'<span style="background:#005AFF;color:#fff;'
                        f'padding:3px 10px;border-radius:4px;font-size:0.9em;font-weight:bold">'
                        f'State {si + 1}</span>',
                        unsafe_allow_html=True,
                    )

                    nkey = f"name_{ridx}_{sidx}_{si}"
                    name_col, name_chk_col = st.columns([6, 1])
                    with name_col:
                        st.markdown(f"**名前：** {state.get('name', '')}")
                    with name_chk_col:
                        state["name_check"] = st.checkbox(
                            "違和感あり",
                            value=state.get("name_check", False),
                            key=f"{nkey}_chk",
                            label_visibility="collapsed",
                        )
                    if state["name_check"]:
                        state["name_comment"] = st.text_area(
                            "コメント",
                            value=state.get("name_comment", ""),
                            key=f"{nkey}_cmt",
                            height=70,
                            label_visibility="collapsed",
                            placeholder="何が間違っている・違和感があるか記入してください",
                        )

                    st.markdown("---")

                    interactions = state.get("utensil_interactions_list", [])
                    if not interactions:
                        st.caption("生成元の記録がありません")

                    for ii, inter in enumerate(interactions):
                        uid = inter.get("_uid", f"{state['id']}_{ii}")
                        wkey = f"u_{ridx}_{sidx}_{si}_{uid}"

                        if ii > 0:
                            st.markdown(
                                "<hr style='margin:10px 0;border:none;border-top:1px solid #ddd'>",
                                unsafe_allow_html=True,
                            )

                        src_col, vessel_col, tools_col = st.columns(3)

                        with src_col:
                            sid = inter.get("source_state_id", "")
                            step, sname, _pos = id_index.get(sid, (0, sid, ""))
                            label = source_label(step, sname) if sid else "（なし）"
                            box_col, chk_col = st.columns([5, 1])
                            with box_col:
                                st.markdown(
                                    '<div style="background:#F0F0F0;border-left:4px solid #9E9E9E;'
                                    'border-radius:4px;padding:6px 10px;margin-bottom:4px">'
                                    f"生成元：<b>{label}</b></div>",
                                    unsafe_allow_html=True,
                                )
                            with chk_col:
                                inter["source_state_id_check"] = st.checkbox(
                                    "違和感あり",
                                    value=inter.get("source_state_id_check", False),
                                    key=f"{wkey}_src_chk",
                                    label_visibility="collapsed",
                                )
                            if inter["source_state_id_check"]:
                                inter["source_state_id_comment"] = st.text_area(
                                    "コメント",
                                    value=inter.get("source_state_id_comment", ""),
                                    key=f"{wkey}_src_cmt",
                                    height=70,
                                    label_visibility="collapsed",
                                )

                        with vessel_col:
                            vessels = inter.get("vessel", [])
                            box_col, chk_col = st.columns([5, 1])
                            with box_col:
                                st.markdown(
                                    '<div style="background:#E3F2FD;border-left:4px solid #4DC4FF;'
                                    'border-radius:4px;padding:6px 10px;margin-bottom:4px">'
                                    "使用容器：<b>"
                                    + ("、".join(vessels) if vessels else "（なし）")
                                    + "</b></div>",
                                    unsafe_allow_html=True,
                                )
                            with chk_col:
                                inter["vessel_check"] = st.checkbox(
                                    "違和感あり",
                                    value=inter.get("vessel_check", False),
                                    key=f"{wkey}_vessel_chk",
                                    label_visibility="collapsed",
                                )
                            if inter["vessel_check"]:
                                inter["vessel_comment"] = st.text_area(
                                    "コメント",
                                    value=inter.get("vessel_comment", ""),
                                    key=f"{wkey}_vessel_cmt",
                                    height=70,
                                    label_visibility="collapsed",
                                )

                        with tools_col:
                            tools = inter.get("tools", [])
                            box_col, chk_col = st.columns([5, 1])
                            with box_col:
                                st.markdown(
                                    '<div style="background:#FFF3E0;border-left:4px solid #F6AA00;'
                                    'border-radius:4px;padding:6px 10px;margin-bottom:4px">'
                                    "使用道具：<b>"
                                    + ("、".join(tools) if tools else "（なし）")
                                    + "</b></div>",
                                    unsafe_allow_html=True,
                                )
                            with chk_col:
                                inter["tools_check"] = st.checkbox(
                                    "違和感あり",
                                    value=inter.get("tools_check", False),
                                    key=f"{wkey}_tools_chk",
                                    label_visibility="collapsed",
                                )
                            if inter["tools_check"]:
                                inter["tools_comment"] = st.text_area(
                                    "コメント",
                                    value=inter.get("tools_comment", ""),
                                    key=f"{wkey}_tools_cmt",
                                    height=70,
                                    label_visibility="collapsed",
                                )


if __name__ == "__main__":
    main()
