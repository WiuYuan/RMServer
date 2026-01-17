import json
from typing import List, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
import textwrap


class LogicalTree:
    class Node:
        def __init__(
            self,
            name: str,
            description: str = "",
            back_num: int = -1,
            parent: Optional["LogicalTree.Node"] = None,
        ):
            self.name = name
            self.description = description
            self.status = "in_progress"  # "in_progress", "success", "fail"
            self.result = None
            self.parent = parent
            self.children: List["LogicalTree.Node"] = []
            self.back_num = back_num

        def add_child(self, child: "LogicalTree.Node"):
            child.parent = self
            self.children.append(child)

        def mark_success(self, result=None):
            self.status = "success"
            self.result = result

        def mark_fail(self, result=None):
            self.status = "fail"
            self.result = result

        def traverse(self):
            yield self
            for child in self.children:
                yield from child.traverse()

        def find_by_name(self, name: str) -> Optional["LogicalTree.Node"]:
            if self.name == name:
                return self
            for child in self.children:
                found = child.find_by_name(name)
                if found:
                    return found
            return None

        def to_dict(self):
            return {
                "name": self.name,
                "description": self.description,
                "status": self.status,
                "result": self.result,
                "children": [child.to_dict() for child in self.children],
            }

        @classmethod
        def from_dict(cls, data: dict, parent: Optional["LogicalTree.Node"] = None):
            node = cls(data["name"], data.get("description", ""), parent)
            node.status = data.get("status", "in_progress")
            node.result = data.get("result")
            for child_data in data.get("children", []):
                child_node = cls.from_dict(child_data, parent=node)
                node.add_child(child_node)
            return node

    def __init__(self, root_name: str, root_description: str = ""):
        self.root = LogicalTree.Node(root_name, root_description)
        self.current_node = self.root
        self.root.back_num = 4

    def add_child(self, name: str, description: str = "") -> str:
        """
        Add a new child node to the current node and set it as the new current node.

        Args:
            name (str): The name of the child node to be created.
            description (str, optional): A short description of the child node.
                Defaults to an empty string.

        Returns:
            LogicalTree.Node: The newly created child node.
        """
        new_node = LogicalTree.Node(
            name, description, back_num=self.current_node.back_num - 1
        )
        self.current_node.add_child(new_node)
        self.current_node = new_node
        return f"Successfully add node with name [{name}]"

    def return_to_parent(self, text=None, success=True) -> str:
        """
        Finalize the current node with a text and move back to its parent node.

        Args:
            text (Any, optional): The result or message describing the outcome
                of the current node. Can include success output or failure details.
            success (bool, optional): Whether the current node is considered
                successful (True) or failed (False). Defaults to True.

        Returns:
            LogicalTree.Node: The parent node of the finalized current node.
                If the current node has no parent, it will remain as the root.
        """
        if success:
            self.current_node.mark_success(text)
        else:
            self.current_node.mark_fail(text)
        if self.current_node.parent is not None:
            self.current_node = self.current_node.parent
        return f"Successfully return node!"

    def add_parent_to_root(self, name: str, description: str = "") -> str:
        """
        Add a new parent node above the current root, making it the new root.

        Args:
            name (str): The name of the new parent node.
            description (str, optional): A short description for the new parent node.
                Defaults to an empty string.

        Returns:
            str: Confirmation message with the new root name.
        """
        new_root = LogicalTree.Node(name, description)
        new_root.add_child(self.root)
        self.root = new_root
        self.current_node = new_root
        return f"Successfully add new root with name [{name}]"

    def find(self, name: str) -> Optional["LogicalTree.Node"]:
        return self.root.find_by_name(name)

    def traverse(self):
        return self.root.traverse()

    def save(self, filepath: str):
        """
        Save the logical tree to a JSON file, including the current node.

        Args:
            filename (str): Path to the file where the tree will be saved.
        """
        tree_dict = self.root.to_dict()
        tree_dict["_current_node"] = self.current_node.name
        with open(filepath, "w") as f:
            json.dump(tree_dict, f, indent=2)

    def to_json(self) -> str:
        """
        Convert the entire logical tree into a JSON-formatted string.

        Returns:
            str: A pretty-printed JSON string representing the tree structure,
                starting from the root node.
        """
        return json.dumps(self.root.to_dict(), indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LogicalTree":
        with open(filepath, "r") as f:
            data = json.load(f)
        tree = cls(data["name"], data.get("description", ""))
        tree.root = LogicalTree.Node.from_dict(data)
        current_node_name = data.get("_current_node", tree.root.name)
        current_node = tree.root.find_by_name(current_node_name)
        tree.current_node = current_node if current_node else tree.root
        return tree

    def visualize(self, max_name_length: int = 15) -> Figure:
        """
        Visualize a LogicalTree object, highlighting the current node
        by adding '*' next to its label and indicating the success/failure status of each node.
        Automatically wraps long names for better display.

        Args:
            tree (LogicalTree): The LogicalTree object (should have current_node and status info).
            max_name_length (int): Maximum characters per line for node names.
        """
        # If the tree was saved, you can call tree.load(json_path) before visualization
        # tree.load("logical_tree.json")

        current_node_name = self.current_node.name

        G = nx.DiGraph()
        node_status = {}  # store each node's status
        node_labels = {}  # store wrapped labels

        # Recursive function to add nodes and edges
        def add_edges(node, parent_name=None):
            G.add_node(node.name)
            node_status[node.name] = getattr(
                node, "status", "pending"
            )  # default pending
            # wrap long names
            wrapped_name = "\n".join(textwrap.wrap(node.name, max_name_length))
            # mark current node
            if node.name == current_node_name:
                wrapped_name += " *"
            node_labels[node.name] = wrapped_name
            if parent_name:
                G.add_edge(parent_name, node.name)
            for child in getattr(node, "children", []):
                add_edges(child, node.name)

        add_edges(self.root)  # start from root

        # Node colors based on status
        color_map = {"success": "lightgreen", "fail": "salmon", "pending": "lightblue"}
        node_colors = [
            color_map.get(node_status.get(n, "pending"), "lightblue") for n in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=2)
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_color=node_colors,
            node_size=2000,
            arrows=True,
            ax=ax,
        )

        for n, (x, y) in pos.items():
            ax.text(x, y, node_labels[n], fontsize=8, ha="center", va="center")

        legend_elements = [
            Patch(facecolor="lightgreen", label="Success"),
            Patch(facecolor="salmon", label="Fail"),
            Patch(facecolor="lightblue", label="Pending"),
        ]
        ax.legend(handles=legend_elements, loc="best")
        ax.set_title("Logical Tree Visualization")

        return fig
