from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import time, json
import networkx as nx
from src.utils.prompts import get_prompt

@dataclass
class NeuronParam:
    instruction: str
    version: int = 0
    history: List[str] = field(default_factory=list)

@dataclass
class NeuronState:
    summary_text: str
    last_update_ts: float

class ThoughtGraph:
    """
    ThoughtGraph
    -------------
    A general graph-based summarization network where each node represents
    a semantic unit with a summarization instruction and a current state.
    Unlike DAG, connections are arbitrary (user-defined).
    """

    DEFAULT_INSTRUCTION = get_prompt("summary_attention_dag/initial_prompt.txt")

    def __init__(self, llm, default_instruction: Optional[str] = None, verbose=True):
        self.G = nx.DiGraph()
        self.llm = llm
        self.default_instruction = default_instruction or self.DEFAULT_INSTRUCTION
        self.verbose = verbose

    # ---------- Node/Edge Management ----------
    def add_node(self, node_id: Any, text: Optional[str] = None, instruction: Optional[str] = None):
        param = NeuronParam(instruction or self.default_instruction)
        state = NeuronState(summary_text=text, last_update_ts=time.time()) if text else None
        self.G.add_node(node_id, param=param, state=state)

    def add_edge(self, src: Any, dst: Any):
        self.G.add_edge(src, dst)

    def remove_edge(self, src: Any, dst: Any):
        if self.G.has_edge(src, dst):
            self.G.remove_edge(src, dst)

    # ---------- Computation ----------
    def recompute_node(self, node_id: Any, question: str):
        """
        Recompute the summary for the given node using:
        - all parent summaries
        - the node's current summary (if any)
        """
        param: NeuronParam = self.G.nodes[node_id]["param"]
        parent_texts = self._collect_parent_texts(node_id)
        current_state: Optional[NeuronState] = self.G.nodes[node_id].get("state")

        prompt = self._build_summary_prompt(
            instruction=param.instruction,
            parent_texts=parent_texts,
            question=question,
            node_id=node_id,
            current_text=current_state.summary_text if current_state else None,
        )
        summary = self.llm.query(prompt, verbose=self.verbose)
        self.G.nodes[node_id]["state"] = NeuronState(
            summary_text=summary,
            last_update_ts=time.time(),
        )


    def _collect_parent_texts(self, node_id: Any) -> List[str]:
        texts = []
        for p in self.G.predecessors(node_id):
            st = self.G.nodes[p].get("state")
            texts.append(st.summary_text if st else "")
        return texts

    # ---------- Instruction update ----------
    def update_instruction(self, node_id: Any, question: str):
        param: NeuronParam = self.G.nodes[node_id]["param"]
        parent_texts = self._collect_parent_texts(node_id)
        prompt = get_prompt("summary_attention_dag/build_param_update_prompt.txt").format(
            mode="GENERAL",
            question=question,
            layer=0, idx=node_id,
            old_instr=param.instruction,
            parents_str="\n".join(parent_texts)
        )
        response = self.llm.query(prompt, verbose=self.verbose)
        param.instruction = response.strip()
        param.version += 1
        param.history.append(f"v{param.version}: {response.strip()}")

    # ---------- Export ----------
    def export_json(self) -> str:
        data = {"nodes": {}, "edges": list(self.G.edges())}
        for n in self.G.nodes:
            p = self.G.nodes[n]["param"]
            s = self.G.nodes[n].get("state")
            data["nodes"][n] = {
                "param": {"instruction": p.instruction, "version": p.version, "history": p.history},
                "state": {"summary_text": s.summary_text if s else None}
            }
        return json.dumps(data, ensure_ascii=False, indent=2)

    # ---------- Prompt builder ----------
    def _build_summary_prompt(
        self,
        instruction: str,
        parent_texts: List[str],
        node_id: Any,
        question: str,
        current_text: Optional[str] = None,
    ) -> str:
        """
        Builds a summarization prompt using both parent summaries and
        the node's current content (if any).
        """
        segs = [f"[Parent {i+1}]\n{t}" for i, t in enumerate(parent_texts)]

        # 新增：当前节点旧内容
        if current_text:
            segs.append(f"[CURRENT STATE]\n{current_text}\n[/CURRENT STATE]")

        segs_str = "\n\n".join(segs) if segs else "[NO INPUT AVAILABLE]"

        return (
            f"You are recomputing the summary for node {node_id}.\n"
            f"Question: {question.replace("\n", "\n\t")}"
            "The following information is available:\n"
            "- Parent summaries (context from predecessors)\n"
            "- The node's current state (previous summary)\n\n"
            f"INPUTS:\n{segs_str.replace("\n", "\n\t")}\n\n"
            f"Goal in this step: {instruction}\n\n"
        )