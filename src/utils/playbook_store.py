from __future__ import annotations

import os
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from fastapi import HTTPException
from pydantic import BaseModel


JsonDict = Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_json_file(p: str) -> JsonDict:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json object: {p}")
    return data


def write_json_file_atomic(p: str, data: JsonDict):
    ensure_dir(os.path.dirname(p))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def list_json_files_flat(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(dir_path)):
        if name.startswith("."):
            continue
        abs_p = os.path.join(dir_path, name)
        if os.path.isfile(abs_p) and name.lower().endswith(".json"):
            out.append(abs_p)
    return out


def list_json_files_recursive(dir_path: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(dir_path):
        return out
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            if fn.lower().endswith(".json"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def safe_relpath(abs_path: str, root: str) -> str:
    rel = os.path.relpath(abs_path, root)
    return rel.replace("\\", "/")


def assert_within_root(abs_path: str, root: str):
    abs_path = os.path.abspath(abs_path)
    root_abs = os.path.abspath(root) + os.sep
    if not abs_path.startswith(root_abs):
        raise HTTPException(status_code=400, detail="Invalid path traversal")


@dataclass(frozen=True)
class PlaybookItemMeta:
    id: str
    title: str
    rel_path: str
    updated_at: Optional[str]
    kind: Literal["policy", "assumption", "backlog"]


class PlaybookStore:
    def __init__(self, playbook_root: str):
        self.root = os.path.abspath(playbook_root)
        self.policies_dir = os.path.join(self.root, "policies")
        self.assumptions_dir = os.path.join(self.root, "assumptions")
        self.backlog_dir = os.path.join(self.root, "backlog")
        self.ensure_structure()

    def ensure_structure(self):
        ensure_dir(self.root)
        ensure_dir(self.policies_dir)
        ensure_dir(self.assumptions_dir)
        ensure_dir(self.backlog_dir)

    def _resolve_abs_path_by_rel(self, rel_path: str) -> str:
        rel = (rel_path or "").lstrip("/").strip()
        abs_path = os.path.abspath(os.path.join(self.root, rel))
        assert_within_root(abs_path, self.root)
        if not abs_path.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="Only .json is supported")
        return abs_path

    def _policy_or_backlog_only_fields(self, data: JsonDict) -> JsonDict:
        return {
            "title": str(data.get("title") or ""),
            "when": str(data.get("when") or ""),
            "next_step": str(data.get("next_step") or ""),
            "notes": str(data.get("notes") or ""),
            "updated_at": utc_now_iso(),
        }

    def _assumption_only_fields(self, data: JsonDict) -> JsonDict:
        return {
            "title": str(data.get("title") or ""),
            "assumption": str(data.get("assumption") or ""),
            "notes": str(data.get("notes") or ""),
            "updated_at": utc_now_iso(),
        }

    def _meta_from_file(self, abs_path: str, kind: Literal["policy", "assumption", "backlog"]) -> PlaybookItemMeta:
        stem = Path(abs_path).stem
        try:
            data = read_json_file(abs_path)
        except Exception:
            data = {}
        title = str(data.get("title") or stem)
        updated_at = data.get("updated_at")
        return PlaybookItemMeta(
            id=stem,
            title=title,
            rel_path=safe_relpath(abs_path, self.root),
            updated_at=updated_at,
            kind=kind,
        )

    def list_policies(self) -> List[PlaybookItemMeta]:
        files = list_json_files_recursive(self.policies_dir)
        return [self._meta_from_file(p, "policy") for p in files]
    
    def get_policy_by_rel(self, rel_path: str) -> JsonDict:
        abs_path = self._resolve_abs_path_by_rel(rel_path)
        # 限制必须在 policies/ 下，避免越区到 backlog/assumptions
        assert_within_root(abs_path, self.policies_dir)

        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail=f"Policy not found: {rel_path}")

        data = read_json_file(abs_path)
        return {
            "title": str(data.get("title") or ""),
            "when": str(data.get("when") or ""),
            "next_step": str(data.get("next_step") or ""),
            "notes": str(data.get("notes") or ""),
            "updated_at": str(data.get("updated_at") or ""),
        }


    def delete_policy_by_rel(self, rel_path: str):
        abs_path = self._resolve_abs_path_by_rel(rel_path)
        assert_within_root(abs_path, self.policies_dir)
        if os.path.exists(abs_path):
            os.remove(abs_path)

    def get_policy(self, policy_id: str) -> JsonDict:
        abs_path = os.path.join(self.policies_dir, f"{policy_id}.json")
        assert_within_root(abs_path, self.root)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")

        data = read_json_file(abs_path)
        return {
            "title": str(data.get("title") or ""),
            "when": str(data.get("when") or ""),
            "next_step": str(data.get("next_step") or ""),
            "notes": str(data.get("notes") or ""),
            "updated_at": str(data.get("updated_at") or ""),
        }

    def upsert_policy(self, rel_dir: str, data: JsonDict, policy_id: Optional[str] = None) -> str:
        rel_dir = (rel_dir or "").strip().strip("/")
        use_id = str(policy_id or data.get("id") or f"P_{uuid.uuid4().hex[:8]}")
        obj = self._policy_or_backlog_only_fields(data)
        if not obj["title"]:
            obj["title"] = use_id

        abs_dir = os.path.abspath(os.path.join(self.policies_dir, rel_dir))
        assert_within_root(abs_dir, self.policies_dir)

        abs_path = os.path.join(abs_dir, f"{use_id}.json")
        assert_within_root(abs_path, self.policies_dir)

        write_json_file_atomic(abs_path, obj)
        return use_id

    def delete_policy(self, policy_id: str):
        abs_path = os.path.join(self.policies_dir, f"{policy_id}.json")
        assert_within_root(abs_path, self.root)
        if os.path.exists(abs_path):
            os.remove(abs_path)

    def list_backlog(self) -> List[PlaybookItemMeta]:
        files = list_json_files_flat(self.backlog_dir)
        return [self._meta_from_file(p, "backlog") for p in files]

    def get_backlog(self, backlog_id: str) -> JsonDict:
        abs_path = os.path.join(self.backlog_dir, f"{backlog_id}.json")
        assert_within_root(abs_path, self.root)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail=f"Backlog not found: {backlog_id}")

        data = read_json_file(abs_path)
        return {
            "title": str(data.get("title") or ""),
            "when": str(data.get("when") or ""),
            "next_step": str(data.get("next_step") or ""),
            "notes": str(data.get("notes") or ""),
            "updated_at": str(data.get("updated_at") or ""),
        }

    def upsert_backlog(self, data: JsonDict, backlog_id: Optional[str] = None) -> str:
        use_id = str(backlog_id or data.get("id") or f"B_{uuid.uuid4().hex[:10]}")
        obj = self._policy_or_backlog_only_fields(data)
        if not obj["title"]:
            obj["title"] = use_id

        abs_path = os.path.join(self.backlog_dir, f"{use_id}.json")
        assert_within_root(abs_path, self.root)
        write_json_file_atomic(abs_path, obj)
        return use_id

    def delete_backlog(self, backlog_id: str):
        abs_path = os.path.join(self.backlog_dir, f"{backlog_id}.json")
        assert_within_root(abs_path, self.root)
        if os.path.exists(abs_path):
            os.remove(abs_path)

    def list_assumptions(self) -> List[PlaybookItemMeta]:
        files = list_json_files_recursive(self.assumptions_dir)
        return [self._meta_from_file(p, "assumption") for p in files]

    def get_assumption(self, rel_path: str) -> JsonDict:
        abs_path = self._resolve_abs_path_by_rel(rel_path)
        assert_within_root(abs_path, self.assumptions_dir)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail=f"Assumption not found: {rel_path}")

        data = read_json_file(abs_path)
        return {
            "title": str(data.get("title") or ""),
            "assumption": str(data.get("assumption") or ""),
            "notes": str(data.get("notes") or ""),
            "updated_at": str(data.get("updated_at") or ""),
        }

    def upsert_assumption(self, rel_dir: str, data: JsonDict, assumption_id: Optional[str] = None) -> str:
        rel_dir = (rel_dir or "").strip().strip("/")
        use_id = str(assumption_id or data.get("id") or f"A_{uuid.uuid4().hex[:8]}")
        obj = self._assumption_only_fields(data)
        if not obj["title"]:
            obj["title"] = use_id

        abs_dir = os.path.abspath(os.path.join(self.assumptions_dir, rel_dir))
        assert_within_root(abs_dir, self.assumptions_dir)

        abs_path = os.path.join(abs_dir, f"{use_id}.json")
        assert_within_root(abs_dir, self.assumptions_dir)
        write_json_file_atomic(abs_path, obj)
        return use_id

    def delete_assumption(self, rel_path: str):
        abs_path = self._resolve_abs_path_by_rel(rel_path)
        assert_within_root(abs_path, self.assumptions_dir)
        if os.path.exists(abs_path):
            os.remove(abs_path)

    def assumptions_tree(self) -> JsonDict:
        return self._build_tree(self.assumptions_dir)

    def _build_tree(self, base_dir: str) -> JsonDict:
        base_dir = os.path.abspath(base_dir)
        assert_within_root(base_dir, self.root)

        def walk(cur_abs: str) -> JsonDict:
            node: JsonDict = {
                "type": "dir",
                "name": os.path.basename(cur_abs),
                "children": []
            }

            try:
                entries = sorted(os.listdir(cur_abs))
            except Exception:
                return node

            dirs: List[str] = []
            files: List[str] = []
            for ent in entries:
                if ent.startswith("."):
                    continue
                abs_p = os.path.join(cur_abs, ent)
                if os.path.isdir(abs_p):
                    dirs.append(abs_p)
                elif os.path.isfile(abs_p) and ent.lower().endswith(".json"):
                    files.append(abs_p)

            for d in dirs:
                node["children"].append(walk(d))

            for f in files:
                meta = self._meta_from_file(f, "assumption")
                node["children"].append({
                    "type": "file",
                    "name": os.path.basename(f),
                    "rel_path": meta.rel_path,
                    "meta": {
                        "id": meta.id,
                        "title": meta.title,
                        "updated_at": meta.updated_at,
                    }
                })

            return node

        ensure_dir(base_dir)
        return walk(base_dir)


class PlaybookListReq(BaseModel):
    playbook_root: str
    kind: Literal["policy", "assumption", "backlog"]
    tree: Optional[bool] = False


class PlaybookGetReq(BaseModel):
    playbook_root: str
    kind: Literal["policy", "assumption", "backlog"]
    id: Optional[str] = None
    rel_path: Optional[str] = None


class PlaybookUpsertPolicyReq(BaseModel):
    playbook_root: str
    rel_dir: str = ""
    policy_id: Optional[str] = None
    data: JsonDict


class PlaybookUpsertBacklogReq(BaseModel):
    playbook_root: str
    backlog_id: Optional[str] = None
    data: JsonDict


class PlaybookUpsertAssumptionReq(BaseModel):
    playbook_root: str
    rel_dir: str = ""
    assumption_id: Optional[str] = None
    data: JsonDict


class PlaybookDeleteReq(BaseModel):
    playbook_root: str
    kind: Literal["policy", "assumption", "backlog"]
    id: Optional[str] = None
    rel_path: Optional[str] = None


async def handle_playbook_list(data: PlaybookListReq) -> JsonDict:
    store = PlaybookStore(data.playbook_root)

    if data.kind == "policy":
        items = store.list_policies()
        return {"ok": True, "kind": "policy", "items": [i.__dict__ for i in items]}

    if data.kind == "backlog":
        items = store.list_backlog()
        return {"ok": True, "kind": "backlog", "items": [i.__dict__ for i in items]}

    if data.kind == "assumption":
        if data.tree:
            return {"ok": True, "kind": "assumption", "tree": store.assumptions_tree()}
        items = store.list_assumptions()
        return {"ok": True, "kind": "assumption", "items": [i.__dict__ for i in items]}

    raise HTTPException(status_code=400, detail="Unknown kind")


async def handle_playbook_get(data: PlaybookGetReq) -> JsonDict:
    store = PlaybookStore(data.playbook_root)

    if data.kind == "policy":
        if data.rel_path:
            return {"ok": True, "kind": "policy", "data": store.get_policy_by_rel(data.rel_path)}
        if not data.id:
            raise HTTPException(status_code=400, detail="Missing id or rel_path for policy")
        return {"ok": True, "kind": "policy", "data": store.get_policy(data.id)}

    if data.kind == "backlog":
        if not data.id:
            raise HTTPException(status_code=400, detail="Missing id for backlog")
        return {"ok": True, "kind": "backlog", "data": store.get_backlog(data.id)}

    if data.kind == "assumption":
        if not data.rel_path:
            raise HTTPException(status_code=400, detail="Missing rel_path for assumption")
        return {"ok": True, "kind": "assumption", "data": store.get_assumption(data.rel_path)}

    raise HTTPException(status_code=400, detail="Unknown kind")


async def handle_playbook_upsert_policy(data: PlaybookUpsertPolicyReq) -> JsonDict:
    store = PlaybookStore(data.playbook_root)
    policy_id = store.upsert_policy(data.rel_dir, data.data, policy_id=data.policy_id)
    return {"ok": True, "policy_id": policy_id}


async def handle_playbook_upsert_backlog(data: PlaybookUpsertBacklogReq) -> JsonDict:
    store = PlaybookStore(data.playbook_root)
    backlog_id = store.upsert_backlog(data.data, backlog_id=data.backlog_id)
    return {"ok": True, "backlog_id": backlog_id}


async def handle_playbook_upsert_assumption(data: PlaybookUpsertAssumptionReq) -> JsonDict:
    store = PlaybookStore(data.playbook_root)
    assumption_id = store.upsert_assumption(
        data.rel_dir,
        data.data,
        assumption_id=data.assumption_id,
    )
    return {"ok": True, "assumption_id": assumption_id}


async def handle_playbook_delete(data: PlaybookDeleteReq) -> JsonDict:
    store = PlaybookStore(data.playbook_root)

    if data.kind == "policy":
        if data.rel_path:
            store.delete_policy_by_rel(data.rel_path)
            return {"ok": True}
        if not data.id:
            raise HTTPException(status_code=400, detail="Missing id or rel_path for policy delete")
        store.delete_policy(data.id)
        return {"ok": True}

    if data.kind == "backlog":
        if not data.id:
            raise HTTPException(status_code=400, detail="Missing id for backlog delete")
        store.delete_backlog(data.id)
        return {"ok": True}

    if data.kind == "assumption":
        if not data.rel_path:
            raise HTTPException(status_code=400, detail="Missing rel_path for assumption delete")
        store.delete_assumption(data.rel_path)
        return {"ok": True}

    raise HTTPException(status_code=400, detail="Unknown kind")