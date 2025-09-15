import os
from typing import Optional
import warnings



# 默认使用 GPT-4o；如需其它模型可在调用时通过 model 参数覆盖。
_DEFAULT_OPENAI_MODEL = "gpt-4o"
_DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# --------------------- Prompt 定义 ----------------------
_PROMPTS = {
    "wipe": (
        "You are filtering a robotics instruction dataset. Keep ONLY instructions that describe exactly ONE 'wipe' action and do NOT mention any other actions. "
        "Good examples: 'Wipe the table with the towel', 'Use the towel to wipe the countertop'. "
        "Bad examples (should be rejected): 'Move the brown object forward then use the towel to wipe the table'.\n\n"
        "Instruction: \"{instruction}\"\n"
        "Answer strictly with 'YES' (keep) or 'NO' (reject)."
    ),
    "pick_place": (
        "You are filtering a robotics instruction dataset. Keep ONLY instructions that describe exactly ONE pick-and-place action pair and do NOT mention any other actions. "
        "Good examples: 'Pick the lid and put it on the pot', 'Pick up the marker from the cup and place it on the table'. "
        "Bad examples (should be rejected): 'Pick up the lid and close the pot', 'Move the black measuring cup to the right then pick up the glass cup and set it down on the desk'.\n\n"
        "Instruction: \"{instruction}\"\n"
        "Answer strictly with 'YES' (keep) or 'NO' (reject)."
    ),
    "single_action": (
        "You are filtering a robotics instruction dataset. Keep ONLY instructions that describe exactly ONE action and do NOT mention any other actions. "
        "The composite action 'pick and place' (or 'pick and put') counts as ONE action. "
        "Good examples: 'Stack the pillows at the edge of the sofa', 'Pick up the pen and put it in the cup' "
        "Bad examples (should be rejected): 'Slide the tap to the center of the sink and press down the tap handle'.\n\n"
        "Instruction: \"{instruction}\"\n"
        "Answer strictly with 'YES' (keep) or 'NO' (reject)."
    ),
}

# --------------------- LLM 调用 ------------------------

import os
from openai import OpenAI

GPT_client = OpenAI(        # 也可以省略，让它自动读环境变量
    api_key=os.getenv("OPENAI_API_KEY"),
)

from google import genai                       # pip install -U google-genai
def _call_gpt(prompt: str,
             model: str = "gpt-4o",   # 或 "gpt-4o-mini"
             temperature: float = 0) -> str | None:
    """
    调用 GPT-4o，返回回答文本。出错时返回 None。
    """
    try:
        resp = GPT_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # v1.x 的返回值是 Pydantic 对象
        return resp.choices[0].message.content.strip()
    except Exception as e:
        warnings.warn(f"OpenAI 调用失败: {e}")
        return None



import os, warnings
from google import genai                       # pip install -U google-genai
from google.genai import types, errors        # 错误类在这里

# 建议把 Client 做成单例，避免反复握手
_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=_api_key) if _api_key else None

def _call_gemini(prompt: str,
                 model: str = "gemini-1.5-flash",
                 temperature: float = 0.7,
                 timeout_sec: int = 15) -> str | None:
    """调用 Gemini（google-genai），失败返回 None。"""
    if gemini_client is None:
        warnings.warn("缺少 GEMINI_API_KEY / GOOGLE_API_KEY")
        return None

    # 1) 通过 GenerateContentConfig 把超时放进 http_options
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        http_options=types.HttpOptions(timeout=timeout_sec * 1000)  # 毫秒! :contentReference[oaicite:0]{index=0}
    )

    try:
        resp = gemini_client.generate_content(
            model=model,
            contents=prompt,     # 字符串会自动转成 user-role Content :contentReference[oaicite:1]{index=1}
            config=cfg,
        )
        return resp.text.strip()
    except errors.APIError as e:               # 正确的错误类位置
        warnings.warn(f"Gemini API error: {e}")
        return None
    except Exception as e:
        warnings.warn(f"Gemini 调用出现未知错误: {e}")
        return None

# --------------------- 公共接口 ------------------------

def instruction_matches_prompt(
    instruction: str,
    *,
    prompt_key: str = "wipe",
    model: str = _DEFAULT_OPENAI_MODEL,
) -> bool:
    """使用指定 prompt 判断指令是否符合要求。

    Args:
        instruction: 待判断的指令。
        prompt_key: 选择使用哪一个 prompt，可为 "wipe" 或 "pick_place"。
        model: GPT 模型名称。

    Returns:
        bool: 指令应被保留返回 True，否则 False。
    """
    instruction = instruction.strip()
    if not instruction:
        return False

    if prompt_key not in _PROMPTS:
        raise ValueError(f"Unknown prompt_key: {prompt_key}. Valid keys: {list(_PROMPTS.keys())}")

    prompt = _PROMPTS[prompt_key].format(instruction=instruction)

    # ----------------- 首选 Gemini -----------------
    llm_answer = _call_gpt(prompt, model=_DEFAULT_OPENAI_MODEL)
    # ----------------- 其次 GPT ----------------
    if llm_answer is None:
        llm_answer = _call_gemini(prompt, model=_DEFAULT_GEMINI_MODEL)

    if llm_answer is None:
        warnings.warn("⚠️  Neither GPT nor Gemini is available; falling back to heuristics.")

    if llm_answer is not None:
        ans_upper = llm_answer.upper()
        if "YES" in ans_upper:
            return True
        if "NO" in ans_upper:
            return False
        # 若回答非预期则回退

    # ---------- 简易回退逻辑 ----------
    # 若 LLM 不可用，则使用 heuristic：根据 prompt_key 粗略判断关键动词数量
    if prompt_key == "wipe":
        return "wipe" in instruction.lower() and not any(v in instruction.lower() for v in ["pick", "place", "move", "grab", "take"])
    elif prompt_key == "pick_place":
        low = instruction.lower()
        return ("pick" in low or "grab" in low) and ("place" in low or "put" in low) and "wipe" not in low and "move" not in low and "close" not in low
    elif prompt_key == "single_action":
        # 粗略启发：出现 "and"/","/"then" 等连接词可能是多动作。允许 'pick and place/put'。
        low = instruction.lower()
        if "pick" in low and ("place" in low or "put" in low):
            # 视为一个复合动作
            return all(word not in low for word in ["wipe", "move", "slide", "stack", "open", "close", "press", "push"]) or True
        # 若包含连接词且不属于允许的 pick-place 形式，则判定为多动作
        if any(conn in low for conn in [" and ", " then ", ","]):
            return False
        # 否则视为单动作
        return True

    return False


# ------------------ 示例用法 -------------------

if __name__ == "__main__":
    demo_instructions = [
        "Wipe the table with the towel",
        "Move the brown object forward then use the towel to wipe the table",
        "Pick the lid and put it on the pot",
        "Slide the tap to the center of the sink and press down the tap handle",
        "Stack the pillows at the edge of the sofa",
    ]

    for key in ["single_action"]: #"wipe", "pick_place"
        print(f"\nPrompt key: {key}")
        for instr in demo_instructions:
            keep = instruction_matches_prompt(instr, prompt_key=key)
            print(f"  [{ 'KEEP' if keep else 'DROP' }] {instr}") 