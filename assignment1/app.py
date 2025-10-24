import streamlit as st
from dotenv import load_dotenv
import os
import requests

# ==== Setup ====
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)

st.set_page_config(page_title="Prompt Optimizer", page_icon="🧠")
st.title("Prompt Improver & Optimizer")

if not GROQ_API_KEY:
    st.error("❌ Falta definir GROQ_API_KEY no .env ou em Streamlit Secrets.")
    st.stop()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

def groq_chat(user_content: str, system_content: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Erro Groq ({resp.status_code}): {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

# ==== UI ====
user_prompt = st.text_area("Write your prompt:", height=200)

if st.button("Optimize Prompt"):
    if not user_prompt.strip():
        st.warning("Please write a prompt first.")
        st.stop()

    try:
        with st.spinner("Structuring your prompt..."):
            structured = groq_chat(
                user_content=(
                    "Given the user's prompt:\n\n"
                    f"{user_prompt}\n\n"
                    "Improve its structure and clarity. Ensure the prompt is complete and unambiguous. "
                    "Return in Markdown with the sections: "
                    "1) Objective  2) Context  3) Constraints  4) Steps/Guidelines  5) Output Format."
                ),
                system_content="You are an expert prompt engineer. Be concise and practical.",
            )

        with st.spinner("Optimizing for LLaMA on Groq..."):
            final = groq_chat(
                user_content=(
                    "Take the structured prompt below and optimize it for a Groq LLaMA model. "
                    "Ensure clarity, token efficiency, and best practices. "
                    "Return ONLY the final prompt text (no explanations).\n\n"
                    f"{structured}"
                ),
                system_content="You optimize prompts for Groq LLaMA models.",
            )

        st.success("Prompt is optimized!")
        st.markdown("**Final Result:**")
        st.markdown(final)

    except Exception as e:
        st.error(f"❌ Error: {e}")
