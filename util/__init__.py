from .agent_tree import Agent, single_chat
from .text_add import add_text, add_text_blocks


def get_top_k_colors(*args, **kwargs):
    from .opposite_color_get import get_top_k_colors as _get_top_k_colors

    return _get_top_k_colors(*args, **kwargs)


def get_opposite_color(*args, **kwargs):
    from .opposite_color_get import get_opposite_color as _get_opposite_color

    return _get_opposite_color(*args, **kwargs)


def __getattr__(name):
    if name in {"LlamaVisionAgent", "MultiAgentManager"}:
        from .llama_agent_util import LlamaVisionAgent, MultiAgentManager

        return {"LlamaVisionAgent": LlamaVisionAgent, "MultiAgentManager": MultiAgentManager}[name]
    raise AttributeError(name)
