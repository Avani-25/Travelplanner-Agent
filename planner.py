"""One-call pipeline: query -> agent -> structured plan -> JSON + readable text."""

from src.agent import run_agent
from src.output import extract_trip_plan, render_text


def plan_trip(agent, llm, query: str) -> dict:
    """Plan a trip and return the structured plan, JSON, readable text and tool steps."""
    run = run_agent(agent, query)
    if run.get("error"):
        return {"error": run["error"]}

    try:
        plan = extract_trip_plan(llm, run["answer"])
    except RuntimeError as exc:
        # Fall back to the agent's own text so the user still gets an answer
        return {"error": str(exc), "agent_answer": run["answer"], "steps": run["steps"]}

    return {
        "plan": plan,
        "text": render_text(plan),
        "json": plan.model_dump_json(indent=2),
        "agent_answer": run["answer"],
        "steps": run["steps"],
    }
