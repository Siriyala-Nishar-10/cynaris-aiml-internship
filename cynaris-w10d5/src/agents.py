"""Optional CrewAI triage and resolution crew; enabled with USE_CREWAI=true."""
import os

def run_support_crew(query: str, context: str) -> str:
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as exc:
        raise RuntimeError('Install requirements.txt to enable CrewAI.') from exc
    model = os.getenv('OLLAMA_MODEL', 'ollama/llama3.2:3b')
    triage = Agent(role='Support Triage Specialist', goal='Classify the customer issue and urgency.', backstory='You carefully triage support requests.', llm=model, verbose=False)
    resolver = Agent(role='Support Resolution Specialist', goal='Draft a helpful response grounded in verified FAQ context.', backstory='You never invent policy or claim actions were completed.', llm=model, verbose=False)
    t1 = Task(description=f'Classify this request and priority: {query}', expected_output='Category and priority.', agent=triage)
    t2 = Task(description=f'Respond to: {query}\nVerified FAQ context:\n{context}\nDo not invent policy.', expected_output='A concise customer support response.', agent=resolver, context=[t1])
    return str(Crew(agents=[triage, resolver], tasks=[t1, t2], process=Process.sequential, verbose=False).kickoff())
