"""The three tasks. `context` passes one task's output to the next, which is how CrewAI agents
share information in a sequential crew."""
from crewai import Task

from research_crew.search import clean_for_prompt


def build_tasks(researcher, writer, reviewer, use_tool=False, web_results=""):
    """Return [research, writing, review].

    use_tool:    tell the researcher to call its web_search tool.
    web_results: search results fetched in advance and given to the researcher as evidence.
    """
    research_text = ("Research the topic: {topic}. Find 5 to 7 key facts, with figures and dates "
                     "where available. After each fact, name its source (title or URL). "
                     "If you cannot verify a fact, say so instead of guessing.")
    if use_tool:
        research_text += (" Use the web_search tool at least twice, with different queries, "
                          "before you answer.")
    if web_results:
        research_text += ("\n\nUse ONLY these web search results as your evidence:\n"
                          + clean_for_prompt(web_results))
    research = Task(
        description=research_text,
        expected_output="A bulleted list of 5 to 7 facts, each followed by its source.",
        agent=researcher,
    )
    writing = Task(
        description=("Using the research notes, write a 400 to 600 word briefing about {topic} "
                     "for a non-technical manager. Every claim must come from the notes. "
                     "Keep the source names or links."),
        expected_output=("A briefing with a title, a short introduction, 3 or 4 short sections "
                         "and a Sources list."),
        agent=writer,
        context=[research],
    )
    review = Task(
        description=("Review the briefing against the research notes. Remove or flag any claim "
                     "the notes do not support, fix unclear wording, and keep the Sources list."),
        expected_output=("The final corrected briefing, followed by 'Changes made:' and a short "
                         "bulleted list of what you changed."),
        agent=reviewer,
        context=[research, writing],
    )
    return [research, writing, review]
