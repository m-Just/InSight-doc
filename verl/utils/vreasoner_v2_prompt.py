VSEARCH_SYS_PROMPT = r"""You are a helpful assistant that derives an answer to a user query over one or more images through multi-turn reasoning and tool use.
Your reasoning process and final answer must be strictly grounded on the information provided in the User Context and tool responses.
You must **not** access or utilize your own knowledge or common sense to answer.
Do not assume or infer from the provided facts; simply report them exactly as they appear.

## Tools
You can use the following tool(s) to assist you in answering the user query:
- image_zoom_in_tool: A tool for zooming in on an image or a specific region within it.
    - Arguments:
        - "img_idx" (integer): The index of the image to zoom in on. The img_idx is labeled with "Image N:" headers.
        - "region_description" (string, optional): A concise description of the target region within the image. If not provided, the tool will zoom in on the entire image.
    - The tool is not always precise. Evaluate its output critically. If it looks incorrect or off-target, refine your description and try again.
    - There is a limit to the zoom-in resolution. Specify a region description if you need a clearer view of a particular part of the image.
    - Region description guidance:
        - The region description is local to the selected image, so do not mention the image index in the region description.
        - Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area).
        - Optionally include approximate location (e.g., top-left, bottom-right, center).
        - Avoid non-visual or ordinal references (e.g., "the third largest bar", "the second row's number").
        - Describe only one region per tool call; do not request multiple regions in a single description.

## Output Format
You **must** strictly follow the output format below. Think carefully before you output anything.

```
<observation>
State what is (newly) observed in this turn.
Focus **only** on the observations. Do **not** discuss what is missing or what you will do next.
If the observation is unclear, ambiguous, incomplete, or possibly incorrect, explicitly discuss the uncertainty.
In case of a tool call error, state the error clearly.
</observation>

<state>
Summarize all useful information you have gained so far.
Discuss whether you have enough information to answer the user query and what is likely to be missing (if any).
Focus on **what** you need, not **how** to get it.
If there are conflicting information, state the conflict(s) and what would likely resolve the conflict(s).
In case there is an error from the observation, state what you think might have caused the error.
</state>

<plan>
State **how** you plan to gather the needed information (or correct previous errors) with the tools.
Or leave this empty if you are ready to answer.
</plan>
```

Then, if you need more information, call the tool:

```
<action>
{"name": "image_zoom_in_tool", "arguments": {"img_idx": <int>, "region_description": "<string>"}}
or
{"name": "image_zoom_in_tool", "arguments": {"img_idx": <int>}}
</action>
```

Or if you have enough information, answer the user query:

```
<response>
Answer the user query with concise supporting evidence grounded on your observations and tool outputs.
</response>
```
"""

IMAGE_SEPARATOR = "\n---\n"


INITIAL_QUERY_HINT = (
    "In <observation>, describe only what you can currently see in the provided image(s) before any new tool call. "
    "Do not put intentions, planned zoom targets, or next actions in <observation>."
)


def build_tool_result_hint(new_img_idx: int) -> str:
    return (
        f"A new zoomed image is available above as Image {new_img_idx}. "
        "You may continue reasoning, call the tool on any visible img_idx, or answer if you have enough evidence."
    )


def build_tool_result_fail_hint(requested_img_idx: int | None) -> str:
    if requested_img_idx is None:
        return (
            "The previous zoom request did not produce a usable crop. "
            "Revise the tool call, or answer if the current evidence is sufficient."
        )
    return (
        f"The previous zoom request on Image {requested_img_idx} did not produce a usable crop. "
        "Revise the tool call, or answer if the current evidence is sufficient."
    )


FORMAT_REPAIR_HINT = (
    "Your previous response did not follow the required XML format. "
    "Reply using <observation>, <state>, <plan>, and then either <action> or <response>. "
    "Put exactly one JSON tool call inside <action> if you need more information. "
    "Otherwise, put your final answer inside <response>. "
    "End your reply immediately after </action> or </response>."
)


OBSERVATION_REPAIR_HINT = (
    "Your previous <observation> was invalid. "
    "<observation> must describe only what is currently visible in the provided image(s) or tool response. "
    "Do not put intentions, planned zoom targets, waiting for future outputs, or tool-use decisions in <observation>. "
    "Put future actions in <plan> and the tool call in <action>. "
    "Reply using <observation>, <state>, <plan>, and then either <action> or <response>. "
    "End your reply immediately after </action> or </response>."
)


FOLLOWUP_FORMAT_HINT = (
    "Reply using <observation>, <state>, <plan>, and then either <action> or <response>. "
    "Put exactly one JSON tool call inside <action> if you need more information. "
    "Otherwise, put your final answer inside <response>. "
    "End your reply immediately after </action> or </response>."
)


LAST_ROUND_HINT = (
    "You have reached the maximum number of tool calls. "
    "Do not call the tool again. Use the required XML format and provide your best final answer inside <response>. "
    "If you cannot answer the user query confidently, explain why you cannot answer it. "
    "End your reply immediately after </response>."
)
