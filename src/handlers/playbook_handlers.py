import json
from src.services.llm import LLM
from src.utils.playbook_store import PlaybookStore
from src.models.requests import PlaybookSuggestNextStepReq

async def handle_playbook_suggest_next_step(data: PlaybookSuggestNextStepReq) -> dict:
    store = PlaybookStore(data.playbook_root)

    # 收集历史 policies（包含 when/next_step 供相似度判断）
    metas = store.list_policies()
    policies = []
    id_to_rel = {}

    for m in metas:
        id_to_rel.setdefault(m.id, []).append(m.rel_path)  # 兼容同名 id（不同文件夹）

        # ✅ 用 rel_path 读取（支持子文件夹）
        p = store.get_policy_by_rel(m.rel_path)

        policies.append({
            "id": m.id,
            "rel_path": m.rel_path,   # ✅ 关键：给 LLM 一个稳定可定位的引用
            "title": p.get("title", ""),
            "when": p.get("when", ""),
            "next_step": p.get("next_step", ""),
        })

    assumptions_text = "\n".join([f"- {a}" for a in (data.assumptions or [])]) or "- (none)"

    prompt = f"""
你是一个决策助手。请基于“假设”和“当前条件”，在给定历史 policy 列表中找到“when 描述最相似”的一条.

要求：
只输出 policy_id, 一个string（输出最相似 policy 的 rel_path；例如 "policies/xxx/P_123.json"；若都不匹配，输出空字符串）
不要输出任何其他东西, 不要输出双引号, 只输出一个rel_path

【假设】
{assumptions_text}

【当前条件】
{data.current_condition}

【历史 policies】（每条都含 id 与 rel_path，优先用 rel_path 做返回值）
{json.dumps(policies, ensure_ascii=False, indent=2)}
""".strip()

    llm = LLM(
        api_key=data.api_key,
        llm_url=data.llm_url,
        model_name=data.model_name,
        format="openai",
        ec=None,
    )

    policy_ref = llm.query(prompt)
    # obj = extract_json_by_braces(raw)

    # ✅ 如果 LLM 直接返回 rel_path（包含 / 或以 .json 结尾），直接用
    if policy_ref and ("/" in policy_ref or policy_ref.lower().endswith(".json")):
        chosen_rel_path = policy_ref
    else:
        # ✅ 否则当作 id，尝试映射到 rel_path（若重复，取第一个）
        cands = id_to_rel.get(policy_ref, [])
        chosen_rel_path = cands[0] if cands else ""

    # ✅ 额外返回：该 policy 文件中存储的原始 next_step（用于对照/溯源）
    policy_next_step = ""
    if chosen_rel_path:
        try:
            p = store.get_policy_by_rel(chosen_rel_path)
            policy_next_step = str(p.get("next_step") or "")
        except Exception:
            policy_next_step = ""

    return {
        "ok": True,
        "policy_id": chosen_rel_path,                 # rel_path
        "policy": policy_next_step,         # policy 原文 next_step
    }
