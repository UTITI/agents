# assignment1/main.py
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

# Carrega variáveis do .env (GROQ_API_KEY, etc.)
load_dotenv()

def main():
    # Checagem rápida para evitar erro de chave ausente
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Faltou definir GROQ_API_KEY no .env (ou no ambiente). "
            "Abra o .env e adicione: GROQ_API_KEY=seu_token_da_groq"
        )

    # ===== 1) Definição do LLM (Groq) =====
    target_model = "groq/llama-3.1-8b-instant"
    llm = LLM(model=target_model)

    # ===== 2) Agentes =====
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

    # ===== 3) Tarefas =====
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
            "Ensure best practices for that model are followed (instruction style, examples if helpful, and token efficiency)."
        ),
        expected_output=(
            "A final optimized prompt ready to be sent to {target_model}. "
            "Return ONLY the final prompt text, no explanations."
        ),
        agent=llm_optimization_agent,
        context=[structure_task],
    )

    # ===== 4) Crew =====
    crew = Crew(
        agents=[prompt_structure_agent, llm_optimization_agent],
        tasks=[structure_task, optimize_task],
    )

    # ===== 5) Input interativo =====
    print("Assignment 1 — Prompt Improver & Optimizer\n")
    try:
        user_prompt = input("👉 Escreve o teu prompt: ").strip()
        if not user_prompt:
            print("Sem prompt. A sair.")
            return

        print(f"\n🧠 Modelo alvo: {target_model}")
        print("🧠 Original prompt:\n", user_prompt)

        result = crew.kickoff(inputs={"user_prompt": user_prompt, "target_model": target_model})

        try:
            print("\n✨ Enhanced & optimized prompt:\n", result.raw)
        except AttributeError:
            print("\n✨ Enhanced & optimized prompt:\n", result)

    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador.")

if __name__ == "__main__":
    main()
