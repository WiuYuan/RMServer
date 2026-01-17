import importlib.resources as pkg_resources
import resources.prompts


from importlib import resources


def get_prompt(prompt_filename: str) -> str:
    """
    支持多层路径: "summary_attention_dag/initial_prompt.txt"
    """
    parts = prompt_filename.split("/")
    if len(parts) == 1:
        pkg = "resources.prompts"
        name = parts[0]
    else:
        pkg = "resources.prompts." + ".".join(parts[:-1])
        name = parts[-1]

    return resources.files(pkg).joinpath(name).read_text(encoding="utf-8")
