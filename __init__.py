from .nodes import *

NODE_CLASS_MAPPINGS = {
    "prompt_sorting": prompt_sorting,
    "apply_lighting_effects": apply_lighting_effects,
    "clean_prompt_tags": clean_prompt_tags,
    "prompt_blacklist": prompt_blacklist,
    "resize_image_sdxl_ratio": resize_image_sdxl_ratio,
    "noline_process": noline_process,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "prompt_sorting": "Auto Prompt Sorting",
    "apply_lighting_effects":"Apply Lighting Effects",
    "clean_prompt_tags":"Clean Prompt Tags",
    "prompt_blacklist": "Prompt Blacklist",
    "resize_image_sdxl_ratio": "SDXL Image Ratio",
    "noline_process": "Noline Process",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
