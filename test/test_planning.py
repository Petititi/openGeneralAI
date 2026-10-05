"""Plan parsing and prompt templates."""

import json

import pytest

from opengeneralai.agent import prompts
from opengeneralai.agent.plan import Plan, PlanDecision, StepStatus, parse_decision, parse_plan
from opengeneralai.errors import PlanParseError


def test_parse_plan_from_fenced_json():
    raw = '```json\n{"plan_steps": [{"descr": "Read", "status": "TODO"}, {"descr": "Fix", "status": "weird"}]}\n```'
    plan = parse_plan(raw)
    assert [s.descr for s in plan.steps] == ["Read", "Fix"]
    assert [s.status for s in plan.steps] == [StepStatus.TODO, StepStatus.TODO], "unknown status -> todo"
    assert plan.current_index() == 0 and not plan.is_done


def test_stop_and_final_end_the_task():
    assert parse_plan("STOP").is_done
    assert parse_plan('FINAL: {"plan_steps": [{"descr": "Nothing to do", "status": "todo"}]}').is_done


@pytest.mark.parametrize("raw", ["not json", '{"steps": []}', '{"plan_steps": [{"status": "todo"}]}', "[1, 2]"])
def test_invalid_plans_are_rejected(raw):
    with pytest.raises(PlanParseError):
        parse_plan(raw)


def test_parse_decision():
    assert parse_decision('{"status": "next_task"}') == PlanDecision.NEXT_TASK
    with pytest.raises(PlanParseError):
        parse_decision('{"status": "MAYBE"}')


def test_description_lists_done_steps_and_the_current_one():
    plan = Plan.from_dict({"plan_steps": [{"descr": "A", "status": "done"}, {"descr": "B"}, {"descr": "C"}]})
    assert plan.description() == "Action plan:\nDone: A\nTODO: B\n"


def test_templates_render_without_doubled_braces():
    plan = Plan.from_dict({"plan_steps": [{"descr": "A", "status": "error"}]})
    rendered = [
        prompts.CREATE_PLAN.format(),
        prompts.NEW_PLAN.format(existing_plan=plan.to_json()),
        prompts.CONTINUE_PLAN.format(task_description="A", next_step_criteria="x", next_task_description="B"),
        prompts.LAST_STEP.format(task_description="A", next_step_criteria="x"),
        prompts.ACTION.format(tool_signatures="read_file(filepath)"),
        prompts.MEMORY_QUERY_ANALYSIS.format(query="q"),
    ]
    for text in rendered:
        assert "{{" not in text and "}}" not in text
    schema = prompts.CREATE_PLAN.format().split("JSON schema:\n")[1]
    assert json.loads(schema)["plan_steps"][0]["status"] == "todo", "the schema shown to the LLM is valid JSON"
    assert '"status": "error"' in rendered[1], "the new plan prompt shows the existing plan"
