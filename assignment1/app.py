import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

# Carrega o .env (GROQ_API_KEY)
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ Falta definir GROQ_API_KEY no .env!")
    st.stop()

llm = LLM(model="groq/llama-3.1-8b-instant")

# === Agents ===
prompt_structure_agent = Agent(
    role="Prompt Structure Agent",
    goal="Improve the structure, clarity, and completeness of user prompts.",
    backstory=(
        "An expert in prompt engineering that rewrites and organizes prompts "
        "to make them clearer and more actionable for AI models."
    ),
    llm=llm,
)

llm_optimization_agent = Agent(
    role="LLM Optimization Agent",
    goal="Optimize prompts for a specific LLM model.",
    backstory=(
        "A specialist in fine-tuning prompts based on model capabilities, "
        "best practices, and token efficiency."
    ),
    llm=llm,
)

# === Tasks ===
structure_task = Task(
    description=(
        "Given the user's prompt:\n\n"
        "{user_prompt}\n\n"
        "Improve its structure and clarity. Ensure the prompt is complete and unambiguous."
    ),
    expected_output=(
        "A rewritten prompt in Markdown with the sections:\n"
        "1) Objective\n2) Context\n3) Constraints\n4) Steps/Guidelines\n5) Output Format"
    ),
    agent=prompt_structure_agent,
)

optimize_task = Task(
    description=(
        "Take the structured prompt produced earlier and optimize it for the target model: {target_model}.\n"
        "Ensure best practices for that model are followed."
    ),
    expected_output=(
        "A final optimized prompt ready to be sent to {target_model}. "
        "Return ONLY the final prompt text, no explanations."
    ),
    agent=llm_optimization_agent,
    context=[structure_task],
)

crew = Crew(
    agents=[prompt_structure_agent, llm_optimization_agent],
    tasks=[structure_task, optimize_task],
)

# === Interface ===
st.set_page_config(page_title="Prompt Optimizer", page_icon="")

st.title("Prompt Improver & Optimizer")

user_prompt = st.text_area("Write your prompt:", height=200)
if st.button("Optimize Prompt"):
    if not user_prompt.strip():
        st.warning("Please write a prompt first.")
        st.stop()

    with st.spinner("Generating and optimizing your prompt"):
        result = crew.kickoff(inputs={
            "user_prompt": user_prompt,
            "target_model": "groq/llama-3.1-8b-instant",
        })

    try:
        final = result.raw
    except AttributeError:
        final = result

    st.success("Prompt is optimized!")
    st.markdown("Final Result:")
    st.markdown(final)
