"""Minimal multi-provider LLM call. Uses the first API key that is set:
Anthropic, OpenAI, Gemini (or PROVIDER=... to force one)."""

import os
import time

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-6-luna",
    "gemini": "gemini-3.6-flash",
}
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def pick_provider() -> tuple[str, str]:
    forced = os.getenv("PROVIDER", "").strip().lower()
    candidates = [forced] if forced else list(KEY_VARS)
    for name in candidates:
        if name not in KEY_VARS:
            raise SystemExit(f"Unknown PROVIDER={forced!r}; use anthropic, openai or gemini.")
        if os.getenv(KEY_VARS[name]):
            return name, os.getenv(f"{name.upper()}_MODEL", DEFAULT_MODELS[name])
    raise SystemExit("No API key found. Copy .env.example to .env and set one of: " + ", ".join(KEY_VARS.values()))


def _call(provider: str, model: str, system: str, user: str) -> str:
    if provider == "anthropic":
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=1024, system=system, messages=[{"role": "user", "content": user}]
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    if provider == "openai":
        import openai

        resp = openai.OpenAI().chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return resp.choices[0].message.content or ""

    from google import genai
    from google.genai import types

    # Keep the client in a `with` block: a temporary Client gets garbage-collected
    # (closing its HTTP session) before the request is sent.
    with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client:
        resp = client.models.generate_content(
            model=model, contents=user, config=types.GenerateContentConfig(system_instruction=system, temperature=0)
        )
    return resp.text or ""


def complete(system: str, user: str, attempts: int = 3) -> str:
    provider, model = pick_provider()
    for attempt in range(1, attempts + 1):
        try:
            return _call(provider, model, system, user).strip()
        except Exception as e:  # rate limits / 503 "high demand" are common on free tiers
            if attempt == attempts:
                raise
            print(f"({provider} error: {e}; retrying)")
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")
