# assignment2/main.py
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

# Carrega variáveis do .env na raiz (GROQ_API_KEY)
load_dotenv()

# Falha cedo se a chave não existir
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "Falta GROQ_API_KEY no .env da raiz (C:\\Users\\...\\workshop-solutions\\.env)."
    )

# ==== LLM (Groq – modelo ativo) ====
# Requer GROQ_API_KEY no .env
llm = LLM(model="groq/llama-3.1-8b-instant")

# ==== Agents ====
content_creation_agent = Agent(
    role="Content Creator",
    goal="Generate engaging and professional posts for the given topic and platform.",
    backstory="An experienced social media manager skilled in creating tailored posts for each platform.",
    llm=llm,
)

safety_check_agent = Agent(
    role="Content Safety Moderator",
    goal="Ensure the generated content is safe, appropriate, and follows community guidelines.",
    backstory="A specialist in moderating and verifying that generated content is compliant and free of harmful language.",
    llm=llm,
)

# ==== Tasks ====
create_content_task = Task(
    description=(
        "Create a social media post about the topic '{topic}' for the '{platform}' platform. "
        "Make it engaging, helpful, clear, and professional. Keep it concise."
    ),
    expected_output=(
        "A well-written post text suitable for the target platform. "
        "Return ONLY the post text, no extra explanations."
    ),
    agent=content_creation_agent,
    output_key="content",  # grava no resultado final
)

# ⚠️ IMPORTANTE: usamos 'context=[create_content_task]' para passar a saída
# da primeira tarefa para a segunda, sem interpolar {content} na description.
safety_check_task = Task(
    description=(
        "Review the previous task's output for safety and appropriateness. "
        "Ensure it has no offensive, violent, discriminatory, medical/legal risk claims, "
        "or sensitive personal data. If issues exist, rewrite a safe version; otherwise, "
        "return the approved text unchanged."
    ),
    expected_output=(
        "A safe, approved version of the content ready for publishing. "
        "Return ONLY the final approved text."
    ),
    agent=safety_check_agent,
    output_key="approved_content",
    context=[create_content_task],  # <- encadeia com a saída da 1ª task
)

# ==== Crew ====
crew = Crew(
    name="Social Media Content Crew",
    agents=[content_creation_agent, safety_check_agent],
    tasks=[create_content_task, safety_check_task],
)

def main():
    topic = input("Enter a topic for the post: ")
    platform = input("Enter the target platform (e.g., LinkedIn, Twitter): ")

    # Apenas 'topic' e 'platform' — a 2ª task recebe o texto via 'context'
    result = crew.kickoff(inputs={"topic": topic, "platform": platform})

    # 'result' pode ser dict (com as chaves definidas por output_key)
    post = result.get("content") if isinstance(result, dict) else result
    approved = result.get("approved_content") if isinstance(result, dict) else None

    print("\n🧠 Generated content:\n", post)
    print("\n🛡️  Approved content:\n", approved or "(no approved version produced)")

if __name__ == "__main__":
    main()
