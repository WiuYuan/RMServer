from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time
import json
import networkx as nx
from src.utils.prompts import get_prompt
import os


@dataclass
class NeuronParam:
    instruction: str
    version: int = 0
    history: List[str] = field(default_factory=list)


@dataclass
class NeuronState:
    summary_text: str
    last_update_ts: float


class SummaryAttentionDAG:
    """
    SummaryAttentionDAG
    -------------------
    A growing DAG where:
      - Layer 0 holds raw input chunks (N grows over time).
      - Layers 1..m mirror size N; neuron i at layer L consumes up to the previous 10 neurons (0..i-1) from layer L-1.
      - Each neuron holds a 'parameter' which is a summarization instruction.
      - Forward pass produces a summary (state) by querying the provided LLM with inputs and instruction.
      - Training updates parameters (instructions) for selected neurons given a user query (question A),
        again via the LLM (minimally-edited new instruction), then (optionally) recomputes the neuron output.

    Requirements:
      - Provide an `llm` object exposing `llm.query(info)` -> str. The `info` is a dict with fields:
          {
            "role": "summarize" | "update_instruction" | "answer",
            "prompt": "<string>",
            "meta": {...}  # optional
          }

    Notes:
      - Uses networkx.DiGraph to store nodes/edges.
      - Node key format: (layer: int, idx: int)
      - Node attributes:
          {
            "param": NeuronParam,
            "state": NeuronState or None
          }
      - Edges connect from (layer-1, j) -> (layer, i) for the parent window j in [i-10, i-1].
    """

    DEFAULT_INSTRUCTION = get_prompt("summary_attention_dag/initial_prompt.txt")

    def __init__(
        self,
        m_layers: int,
        llm: Any,
        default_instruction: Optional[str] = None,
        parent_window: int = 10,
        verbose: bool = True
    ):
        assert m_layers >= 1, "m_layers must be >= 1"
        self.m_layers = m_layers
        self.llm = llm
        self.parent_window = parent_window
        self.default_instruction = default_instruction or self.DEFAULT_INSTRUCTION

        # Graph init
        self.G = nx.DiGraph()
        # Track current N (number of chunks / width)
        self.N = 0
        self.verbose = verbose

    # --------------------------- Public API ---------------------------
    def add_chunk(self, chunk_text: str) -> None:
        """
        Append a new raw chunk to layer 0 and build corresponding nodes in layers 1..m incrementally.
        """
        idx = self.N  # new index
        # Layer 0 node with raw text as state summary
        self._add_node(
            layer=0,
            idx=idx,
            param=NeuronParam(instruction="LEAF_RAW"),
            state=NeuronState(summary_text=chunk_text, last_update_ts=time.time()),
        )

        # Build layers 1..m for this index
        for layer in range(1, self.m_layers + 1):
            self._build_and_compute_node(layer=layer, idx=idx)

        self.N += 1
        
    def _apply_instruction_to_all_nodes(self, new_instruction: str) -> None:
        """
        将新的 instruction 强制应用到所有非叶子节点（layer>=1）。
        """
        for (layer, idx), data in self.G.nodes(data=True):
            if layer == 0:
                continue  # leaf nodes skip
            param: NeuronParam = data["param"]
            param.history.append(
                f"GLOBAL_UPDATE v{param.version}->{param.version+1}"
            )
            param.version += 1
            param.instruction = new_instruction

    def retrain_params_for_query(
        self,
        question: str,
        top_layer: Optional[int] = None,
        k_tail: int = 1,
        recompute: bool = True,
        propagate_upstream: bool = False,
        apply_to_all: bool = False,
    ) -> List[Tuple[Tuple[int, int], str]]:
        """
        Given a user query (question), update parameters for the last k_tail neurons on the top_layer.
        Returns a list of ((layer, idx), new_instruction).

        Args:
          question: the question A used to tailor the node instructions.
          top_layer: which layer to train on (defaults to highest layer m).
          k_tail: how many of the most recent indices to update (default 1).
          recompute: whether to recompute summaries after updating instruction.
          propagate_upstream: if True, also proposes light updates to direct parents' instructions.
        """
        L = self.m_layers if top_layer is None else top_layer
        results = []
        # Determine tail indices that exist on that layer
        tail_indices = [i for i in range(max(0, self.N - k_tail), self.N)]
        for idx in tail_indices:
            node_key = (L, idx)
            if not self.G.has_node(node_key):
                continue
            new_instr, rationale = self._update_instruction_for_node(node_key, question)
            if new_instr is not None:
                # apply
                param: NeuronParam = self.G.nodes[node_key]["param"]
                param.history.append(
                    f"v{param.version}->{param.version+1}: {rationale}"
                )
                param.version += 1
                param.instruction = new_instr
                results.append((node_key, new_instr))
                if recompute:
                    self._recompute_node_state(node_key)
                    
                if apply_to_all:
                    if self.verbose:
                        print("[INFO] Applying updated instruction to ALL neurons...")
                    self._apply_instruction_to_all_nodes(new_instr)

                    if recompute:
                        if self.verbose:
                            print("[INFO] Recomputing ALL states...")
                        self.recompute_all_states()

                if propagate_upstream:
                    # Lightly propose updates to direct parents
                    for p in self._parents(node_key):
                        p_instr, p_rat = self._update_instruction_for_node(
                            p, question, upstream_hint=True
                        )
                        if p_instr is not None:
                            p_param: NeuronParam = self.G.nodes[p]["param"]
                            p_param.history.append(
                                f"UPSTREAM v{p_param.version}->{p_param.version+1}: {p_rat}"
                            )
                            p_param.version += 1
                            p_param.instruction = p_instr
                            if recompute and p[0] > 0:
                                # Do not recompute L0 leaves
                                self._recompute_node_state(p)

        return results
    
    def _inherit_instruction(self, layer: int, idx: int) -> str:
        """
        返回该层前一个节点的 instruction，如果不存在则使用 default_instruction。
        """
        if idx > 0:
            prev_key = (layer, idx - 1)
            if self.G.has_node(prev_key):
                prev_param: NeuronParam = self.G.nodes[prev_key]["param"]
                return prev_param.instruction

        # 否则用 default
        return self.default_instruction

    def answer(
        self, question: str, top_k_nodes: int = 3, top_layer: Optional[int] = None
    ) -> str:
        """
        Generate an answer using top_layer's most recent top_k_nodes summaries, merged by the LLM.
        """
        L = self.m_layers if top_layer is None else top_layer
        idxs = list(range(max(0, self.N - top_k_nodes), self.N))
        segments = []
        for i in idxs:
            node_key = (L, i)
            if self.G.has_node(node_key):
                st: NeuronState = self.G.nodes[node_key].get("state")
                if st:
                    segments.append((i, st.summary_text))

        prompt = self._build_answer_prompt(question, segments)
        info = {
            "role": "answer",
            "prompt": prompt,
            "meta": {"layer": L, "indices": idxs},
        }
        return self.llm.query(info["prompt"], verbose=self.verbose)

    def export_json(self) -> str:
        """
        Export the whole graph (parameters & states) as JSON for audit/backup.
        """
        data = {
            "m_layers": self.m_layers,
            "N": self.N,
            "parent_window": self.parent_window,
            "default_instruction": self.default_instruction,
            "nodes": {},
            "edges": list(self.G.edges()),
        }
        for n in self.G.nodes:
            param: NeuronParam = self.G.nodes[n]["param"]
            state: Optional[NeuronState] = self.G.nodes[n].get("state")
            data["nodes"][str(n)] = {
                "param": {
                    "instruction": param.instruction,
                    "version": param.version,
                    "history": param.history,
                },
                "state": {
                    "summary_text": state.summary_text if state else None,
                    "last_update_ts": state.last_update_ts if state else None,
                },
            }
        return json.dumps(data, ensure_ascii=False, indent=2)

    # --------------------------- Internal helpers ---------------------------
    def _add_node(
        self, layer: int, idx: int, param: NeuronParam, state: Optional[NeuronState]
    ) -> None:
        key = (layer, idx)
        self.G.add_node(key, param=param, state=state)

        # ---- 新增：连接上一层前 10 个节点 ----
        if layer > 0:
            for p in self._parent_keys(layer, idx):
                self.G.add_edge(p, key)

        # ---- 新增：连接本层前一个节点 ----
        if idx > 0:
            prev_key = (layer, idx - 1)
            if self.G.has_node(prev_key):
                self.G.add_edge(prev_key, key)

    def _build_and_compute_node(self, layer: int, idx: int) -> None:
        key = (layer, idx)
        instr = self._inherit_instruction(layer, idx)
        param = NeuronParam(instruction=instr)
        self._add_node(layer, idx, param, state=None)
        # compute
        self._recompute_node_state(key)

    def _recompute_node_state(self, node_key: Tuple[int, int]) -> None:
        layer, idx = node_key
        if layer == 0:
            return  # leaves already have raw state
        param: NeuronParam = self.G.nodes[node_key]["param"]
        parent_texts = self._collect_parent_texts(node_key)
        prompt = self._build_summary_prompt(param.instruction, parent_texts, layer, idx)
        info = {
            "role": "summarize",
            "prompt": prompt,
            "meta": {"layer": layer, "idx": idx},
        }
        summary = self.llm.query(info["prompt"], verbose=self.verbose)
        self.G.nodes[node_key]["state"] = NeuronState(
            summary_text=summary, last_update_ts=time.time()
        )

    def _collect_parent_texts(self, node_key: Tuple[int, int]) -> List[str]:
        texts = []
        for p in self._parents(node_key):
            st: NeuronState = self.G.nodes[p].get("state")
            if st:
                texts.append(st.summary_text)
            else:
                texts.append("")
        return texts

    def _parent_keys(self, layer: int, idx: int) -> List[Tuple[int, int]]:
        assert layer > 0
        start = max(0, idx - self.parent_window)
        return [(layer - 1, j) for j in range(start, idx + 1)]

    def _parents(self, node_key: Tuple[int, int]) -> List[Tuple[int, int]]:
        return [p for p, _ in self.G.in_edges(node_key)]

    def _update_instruction_for_node(
        self, node_key: Tuple[int, int], question: str, upstream_hint: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        param: NeuronParam = self.G.nodes[node_key]["param"]
        parent_texts = self._collect_parent_texts(node_key)
        prompt = self._build_param_update_prompt(
            old_instr=param.instruction,
            parent_summaries=parent_texts,
            question=question,
            layer=node_key[0],
            idx=node_key[1],
            upstream_hint=upstream_hint,
        )
        info = {
            "role": "update_instruction",
            "prompt": prompt,
            "meta": {"node": node_key},
        }
        response = self.llm.query(info["prompt"], verbose=self.verbose)
        try:
            # Expecting a small JSON or delimiter-based response; allow fallback parsing
            parsed = self._parse_update_response(response)
            new_instr = parsed.get("NEW_INSTRUCTION")
            rationale = parsed.get("RATIONALE", "")
            if (
                new_instr
                and new_instr.strip()
                and new_instr.strip() != param.instruction.strip()
            ):
                return new_instr.strip(), rationale.strip()
        except Exception:
            # If parsing fails, treat whole response as new instruction string
            if (
                response
                and response.strip()
                and response.strip() != param.instruction.strip()
            ):
                return response.strip(), "freeform-update"
        return None, None

    # --------------------------- Prompt builders ---------------------------
    def _build_summary_prompt(
        self, instruction: str, parent_texts: List[str], layer: int, idx: int
    ) -> str:
        segments = []
        for j, t in enumerate(parent_texts, 1):
            segments.append(f"<<<SEGMENT {j} START>>>\n{t}\n<<<SEGMENT {j} END>>>")
        segments_str = "\n\n".join(segments) if segments else "[NO PARENT CONTENT]"

        prompt = (
            "You are a summarizer that must preserve chronology and factuality.\n"
            f"NODE: Layer={layer}, Index={idx}\n"
            f"GOAL (instruction): {instruction}\n\n"
            "CONSTRAINTS:\n"
            "- Preserve temporal order strictly.\n"
            "- Keep named entities and coreference consistent with sources.\n"
            "- No fabrication; if uncertain, mark as [UNCERTAIN].\n"
            "- Hard max length: 10,000 characters.\n\n"
            "INPUT SEGMENTS (ordered, oldest -> newest):\n"
            f"{segments_str}\n\n"
            "TASK:\n"
            "1) Merge the segments into ONE coherent summary.\n"
            "2) Maintain a clear timeline and causality.\n"
            "3) Finish with 3-5 bullet points of Key Facts.\n"
            "OUTPUT:\n"
        )
        return prompt

    def _build_param_update_prompt(
        self,
        old_instr: str,
        parent_summaries: List[str],
        question: str,
        layer: int,
        idx: int,
        upstream_hint: bool = False,
    ) -> str:
        segs = []
        for j, t in enumerate(parent_summaries, 1):
            segs.append(f"<<<PARENT {j}>>>\n{t}")
        parents_str = "\n\n".join(segs) if segs else "[NO PARENTS]"
        mode = "UPSTREAM_HINT" if upstream_hint else "TARGET_NODE"
        prompt = get_prompt(
            "summary_attention_dag/build_param_update_prompt.txt"
        ).format(
            mode=mode,
            question=question.replace("\n", "\t\n"),
            layer=layer,
            idx=idx,
            old_instr=old_instr.replace("\n", "\n\t"),
            parents_str=parents_str.replace("\n", "\n\t"),
        )
        return prompt

    def _build_answer_prompt(
        self, question: str, segments: List[Tuple[int, str]]
    ) -> str:
        segs = []
        for idx, text in segments:
            segs.append(f"<<<TOPLAYER NODE {idx}>>>\n{text}")
        segs_str = "\n\n".join(segs) if segs else "[NO CONTENT]"
        prompt = get_prompt("summary_attention_dag/build_answer_prompt.txt").format(
            question=question.replace("\n", "\n\t"),
            segs_str=segs_str.replace("\n", "\n\t"),
        )
        return prompt

    # --------------------------- Parsers ---------------------------
    def _parse_update_response(self, text: str) -> Dict[str, str]:
        """
        Tries to parse a JSON payload with NEW_INSTRUCTION and RATIONALE.
        Falls back to simple heuristics.
        """
        text = text.strip()
        # Try direct JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return {
                    "NEW_INSTRUCTION": str(obj.get("NEW_INSTRUCTION", "")).strip(),
                    "RATIONALE": str(obj.get("RATIONALE", "")).strip(),
                }
        except Exception:
            pass

        # Try extracting between markers if present
        if "NEW_INSTRUCTION" in text:
            # naive fallback
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            new_instr = ""
            rationale = ""
            for i, l in enumerate(lines):
                if l.lower().startswith("new_instruction"):
                    colon = l.find(":")
                    if colon != -1:
                        new_instr = l[colon + 1 :].strip().strip('"')
                if l.lower().startswith("rationale"):
                    colon = l.find(":")
                    if colon != -1:
                        rationale = l[colon + 1 :].strip()
            return {"NEW_INSTRUCTION": new_instr, "RATIONALE": rationale}

        # Otherwise, treat entire text as NEW_INSTRUCTION
        return {"NEW_INSTRUCTION": text, "RATIONALE": ""}

    def get_last_neuron_value(self, layer: int = None) -> Optional[str]:
        """
        返回指定层最后一个节点的 summary_text。
        若 layer 未指定，则默认取最高层（m_layers）。
        """
        L = self.m_layers if layer is None else layer
        if self.N == 0:
            return None

        last_idx = self.N - 1
        node_key = (L, last_idx)

        if not self.G.has_node(node_key):
            return None

        state = self.G.nodes[node_key].get("state")
        return state.summary_text if state else None

    def recompute_all_states(self) -> None:
        """
        Recompute the summary_text (state) for all nodes except layer 0 leaves.
        This does NOT change parameters (instructions).
        """
        total = 0
        # 按层顺序计算，从 layer=1 到最高层
        for layer in range(1, self.m_layers + 1):
            # 按 index 顺序更新，保证时间依赖一致
            for idx in range(self.N):
                node_key = (layer, idx)
                if not self.G.has_node(node_key):
                    continue
                self._recompute_node_state(node_key)
                total += 1
                if self.verbose:
                    print(f"[RECOMPUTE] Layer {layer}, idx {idx} updated.")

        if self.verbose:
            print(f"[INFO] Recomputed {total} nodes in total (excluding layer 0).")
            
    def save_all_nodes_to_files(self, base_dir: str = "./neuron_dump") -> None:
        """
        将所有神经元节点（参数 + 状态）分别保存为独立 JSON 文件。
        文件名格式：layer_{L}_idx_{i}.json
        """
        os.makedirs(base_dir, exist_ok=True)
        count = 0

        for (layer, idx), data in self.G.nodes(data=True):
            param: NeuronParam = data["param"]
            state: Optional[NeuronState] = data.get("state")

            node_info = {
                "layer": layer,
                "index": idx,
                "instruction": param.instruction,
                "version": param.version,
                "history": param.history,
                "summary_text": state.summary_text if state else None,
                "last_update_ts": state.last_update_ts if state else None,
            }

            path = os.path.join(base_dir, f"layer_{layer}_idx_{idx}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(node_info, f, ensure_ascii=False, indent=2)
            count += 1

        if self.verbose:
            print(f"[SAVE] Save {count} nodes to {base_dir}.")
