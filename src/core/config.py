"""
Lab 11 — Configuration & API Key Setup
"""
import os


def setup_api_key():
    """Load an OpenRouter API key from ``.env`` or prompt for it once."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # Environment variables still work if the optional loader is absent.
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        api_key = input("Enter OpenRouter API Key: ").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for live model calls.")
        os.environ["OPENROUTER_API_KEY"] = api_key
    print("OpenRouter API key loaded.")


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
