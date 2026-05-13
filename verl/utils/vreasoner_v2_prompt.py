VSEARCH_SYS_PROMPT = (
    "You are a visual assistant. Your goal is to answer a question based on one or more images.\n"
    "\n"
    "First, think step by step to identify which visual facts you need from the images to answer the question. "
    "If the visual information is insufficient or unclear, call the visual search tool by specifying a region "
    "description and the index of the image to search:\n"
    "<tool_call>{\"region_description\": \"...\", \"img_idx\": N}</tool_call>\n"
    "\n"
    "The tool will zoom in on the specified region of the selected image and return a new image with a new img_idx. "
    "You may repeat this process on any available image until you have enough evidence to answer confidently. "
    "The tool is not always precise \u2014 evaluate its output critically. If it looks incorrect or off-target, "
    "refine your description and try again.\n"
    "\n"
    "There is a limit to the zoom-in resolution. If an image is already near the maximum zoom level, "
    "the tool may reject the request.\n"
    "\n"
    "Region description guidance:\n"
    "- The region description is local to the selected image, so do not mention the image index in the region description.\n"
    "- Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area)\n"
    "- Optionally include approximate location (e.g., top-left, bottom-right, center)\n"
    "- Avoid non-visual or ordinal references (e.g., \"the third largest bar\", \"the second row's number\")\n"
    "- Describe only one region per tool call; do not request multiple regions in a single description\n"
    "\n"
    "Output format:\n"
    "- Put your reasoning process inside <think>...</think>.\n"
    "- When you need to call the tool, provide the region description and image index using the format "
    "<tool_call>{\"region_description\": \"...\", \"img_idx\": N}</tool_call>.\n"
    "- Immediately after each </think>, do exactly one of:\n"
    "  1) Call the tool; or\n"
    "  2) Provide the final answer (no tool call) \u2014 wrap it in <answer>...</answer>. "
    "Do not mix tool calls and answers in the same turn.\n"
    "You **must strictly follow the output format**, otherwise your answer will be judged as wrong.\n"
    "\n"
    "A multi-turn format example:\n"
    "Assistant:\n"
    "<think>{your step-by-step analysis; decide if more detail is needed}</think>\n"
    "<tool_call>{\"region_description\": \"concise, visually grounded target (optionally with location)\", \"img_idx\": 0}</tool_call>\n"
    "\n"
    "User:\n"
    "[Zoomed-in image labeled with new img_idx + guidance]\n"
    "\n"
    "Assistant:\n"
    "<think>{updated analysis based on the zoomed-in view; decide whether to refine or answer}</think>\n"
    "<tool_call>{\"region_description\": \"next concise target (optionally with location)\", \"img_idx\": 2}</tool_call>\n"
    "\n"
    "(Repeat the User \u2192 Assistant pattern as needed until enough evidence is gathered.)\n"
    "\n"
    "Assistant (final turn):\n"
    "<think>{final reasoning; explain why the available visual evidence is sufficient}</think>\n"
    "<answer>...</answer>"
)


UNANSWERABLE_AWARE_VSEARCH_SYS_PROMPT = (
    "You are a visual assistant. Your goal is to answer a question based on one or more images.\n"
    "\n"
    "First, think step by step to identify which visual facts you need from the images to answer the question. "
    "A question may sometimes be unanswerable, underspecified, or based on a premise that is not supported by the images. "
    "However, do not give up early. You should still search carefully for the needed evidence before concluding that the "
    "question cannot be answered as asked.\n"
    "\n"
    "If the visual information is insufficient or unclear, call the visual search tool by specifying a region "
    "description and the index of the image to search:\n"
    "<tool_call>{\"region_description\": \"...\", \"img_idx\": N}</tool_call>\n"
    "\n"
    "The tool will zoom in on the specified region of the selected image and return a new image with a new img_idx. "
    "You may repeat this process on any available image until you have enough evidence to answer confidently. "
    "The tool is not always precise - evaluate its output critically. If it looks incorrect or off-target, "
    "refine your description and try again.\n"
    "\n"
    "There is a limit to the zoom-in resolution. If an image is already near the maximum zoom level, "
    "the tool may reject the request.\n"
    "\n"
    "How to handle hard questions:\n"
    "- If the images contain enough evidence, answer directly and concretely.\n"
    "- If the needed information is not visible or cannot be determined from the images, say so clearly.\n"
    "- If the question assumes something that the images do not support, say that the premise is unsupported by the images.\n"
    "- Do not invent missing details. Keep the answer grounded in what is actually shown.\n"
    "\n"
    "Region description guidance:\n"
    "- The region description is local to the selected image, so do not mention the image index in the region description.\n"
    "- Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area)\n"
    "- Optionally include approximate location (e.g., top-left, bottom-right, center)\n"
    "- Avoid non-visual or ordinal references (e.g., \"the third largest bar\", \"the second row's number\")\n"
    "- Describe only one region per tool call; do not request multiple regions in a single description\n"
    "\n"
    "Output format:\n"
    "- Put your reasoning process inside <think>...</think>.\n"
    "- When you need to call the tool, provide the region description and image index using the format "
    "<tool_call>{\"region_description\": \"...\", \"img_idx\": N}</tool_call>.\n"
    "- Immediately after each </think>, do exactly one of:\n"
    "  1) Call the tool; or\n"
    "  2) Provide the final answer (no tool call) - wrap it in <answer>...</answer>. "
    "Do not mix tool calls and answers in the same turn.\n"
    "You must strictly follow the output format, otherwise your answer will be judged as wrong.\n"
    "\n"
    "A multi-turn format example:\n"
    "Assistant:\n"
    "<think>{your step-by-step analysis; decide if more detail is needed}</think>\n"
    "<tool_call>{\"region_description\": \"concise, visually grounded target (optionally with location)\", \"img_idx\": 0}</tool_call>\n"
    "\n"
    "User:\n"
    "[Zoomed-in image labeled with new img_idx + guidance]\n"
    "\n"
    "Assistant:\n"
    "<think>{updated analysis based on the zoomed-in view; decide whether to refine or answer}</think>\n"
    "<tool_call>{\"region_description\": \"next concise target (optionally with location)\", \"img_idx\": 2}</tool_call>\n"
    "\n"
    "(Repeat the User -> Assistant pattern as needed until enough evidence is gathered.)\n"
    "\n"
    "Assistant (final turn):\n"
    "<think>{final reasoning; either identify the answer or explain why the images do not support one}</think>\n"
    "<answer>...</answer>"
)


def get_vsearch_sys_prompt(prompt_variant: str | None = None) -> str:
    variant = (prompt_variant or "default").strip().lower()
    if variant == "default":
        return VSEARCH_SYS_PROMPT
    if variant in {"unanswerable_aware", "non_answerable", "not_answerable"}:
        return UNANSWERABLE_AWARE_VSEARCH_SYS_PROMPT
    raise ValueError(f"Unknown vreasoner_v2 prompt variant: {prompt_variant}")


IMAGE_SEPARATOR = "\n---\n"


def build_tool_result_hint(new_img_idx: int) -> str:
    return (
        f"Based on your description, here is the zoomed-in image (Image {new_img_idx}).\n\n"
        "Please continue your analysis. You may:\n"
        "- Call the tool again on any available image if you believe more visual detail is needed; or\n"
        "- Provide your final answer if the current information is sufficient."
    )


def build_tool_result_fail_hint(requested_img_idx: int | None) -> str:
    if requested_img_idx is None:
        return (
            "The visual searcher could not locate the requested target based on your description.\n\n"
            "Please adjust or refine your region description and continue your analysis. You may:\n"
            "- Call the tool again with a revised description; or\n"
            "- Provide your final answer if the current information is sufficient."
        )
    return (
        f"The visual searcher could not locate the requested target in Image {requested_img_idx} "
        "based on your description.\n\n"
        "Please adjust or refine your region description and continue your analysis. You may:\n"
        "- Call the tool again with a revised description; or\n"
        "- Provide your final answer if the current information is sufficient."
    )


FORMAT_REPAIR_HINT = (
    "In your previous response, neither a tool call nor a final answer wrapped in <answer>...</answer> was provided "
    "(or the format is incorrect).\n\n"
    "Please do exactly one of the following:\n"
    "- If you still need more visual detail, call the tool using the exact JSON format:\n"
    "  <tool_call>{\"region_description\": \"...\", \"img_idx\": N}</tool_call>\n"
    "- Otherwise, provide the final answer now using <answer>...</answer>."
)


LAST_ROUND_HINT = (
    "You have reached the limit for using the visual tool and cannot call it again.\n"
    "In this turn, based on the available information, provide your final answer using <answer>...</answer>."
)
