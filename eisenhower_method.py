"""Streamlit Eisenhower Matrix planner application."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_sortables import sort_items


@dataclass
class Task:
    """Represents a single task in the Eisenhower matrix."""

    id: str
    title: str
    description: str
    duration_hours: float
    quadrant: str
    position: int
    created_at: str

    @staticmethod
    def from_dict(raw: Dict) -> "Task":
        """Create a task from a raw dictionary, applying defaults where needed."""

        task_id = str(raw.get("id") or uuid4())
        title = str(raw.get("title") or "Untitled task")
        description = str(raw.get("description") or "")
        try:
            duration = float(raw.get("duration_hours", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        quadrant = raw.get("quadrant") or infer_quadrant(raw)
        if quadrant not in QUADRANTS:
            quadrant = "Q1"
        try:
            position = int(raw.get("position", 0))
        except (TypeError, ValueError):
            position = 0
        created_raw = raw.get("created_at")
        if created_raw:
            try:
                # Validate timestamp
                datetime.fromisoformat(created_raw)
                created_at = created_raw
            except ValueError:
                created_at = datetime.utcnow().isoformat()
        else:
            created_at = datetime.utcnow().isoformat()
        return Task(
            id=task_id,
            title=title,
            description=description,
            duration_hours=max(duration, 0.0),
            quadrant=quadrant,
            position=position,
            created_at=created_at,
        )

    def to_serializable(self) -> Dict:
        data = asdict(self)
        data["duration_hours"] = float(self.duration_hours)
        data["position"] = int(self.position)
        return data


QUADRANTS: Dict[str, Dict[str, str]] = {
    "Q1": {"header": "Do First", "subtitle": "Urgent & Important", "color": "#d9534f"},
    "Q2": {"header": "Schedule", "subtitle": "Not Urgent & Important", "color": "#f0ad4e"},
    "Q3": {"header": "Delegate", "subtitle": "Urgent & Not Important", "color": "#5bc0de"},
    "Q4": {"header": "Eliminate", "subtitle": "Not Urgent & Not Important", "color": "#5cb85c"},
}
HEADER_TO_QUADRANT = {meta["header"]: key for key, meta in QUADRANTS.items()}
PRIORITY_ORDER = ["Q1", "Q2", "Q3", "Q4"]
DEFAULT_SETTINGS = {
    "available_hours": 8.0,
    "break_minutes": 30.0,
    "focus_block_minutes": 120.0,
    "session_start": datetime.combine(datetime.today(), datetime.min.time()).replace(hour=9, minute=0, second=0, microsecond=0),
}
STYLE_BOARD = """
.sortable-component {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
}
.sortable-container {
    flex: 1 1 calc(50% - 1.2rem);
    min-width: 280px;
    background: #f7f9fc;
    border-radius: 12px;
    padding: 0.5rem 0.75rem 1rem 0.75rem;
    border: 1px solid #e5e9f2;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
}
.sortable-container-header {
    font-weight: 700;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.2rem;
    padding: 0.5rem 0.25rem 0.25rem 0.25rem;
    color: #1f2933;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.sortable-container-body {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.sortable-item {
    background: white;
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    border: 1px solid #d0d7e2;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
    white-space: pre-wrap;
    line-height: 1.35;
}
.sortable-item.dragging {
    opacity: 0.7;
}
@media (max-width: 900px) {
    .sortable-container {
        flex: 1 1 100%;
    }
}
"""


def infer_quadrant(raw: Dict) -> str:
    importance = raw.get("importance")
    urgency = raw.get("urgency")
    if importance in {True, "important", "Important", "high", 1, "1"}:
        important = True
    else:
        important = False
    if urgency in {True, "urgent", "Urgent", "high", 1, "1"}:
        urgent = True
    else:
        urgent = False
    if important and urgent:
        return "Q1"
    if important and not urgent:
        return "Q2"
    if urgent and not important:
        return "Q3"
    return "Q4"


def init_session_state() -> None:
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    if "settings" not in st.session_state:
        st.session_state.settings = DEFAULT_SETTINGS.copy()
    else:
        defaults = DEFAULT_SETTINGS.copy()
        defaults.update(st.session_state.settings)
        st.session_state.settings = defaults


def make_task(title: str, description: str, duration: float, quadrant: str) -> Task:
    task_id = str(uuid4())
    existing_positions = [task.position for task in st.session_state.tasks if isinstance(task, Task) and task.quadrant == quadrant]
    next_position = max(existing_positions, default=-1) + 1
    return Task(
        id=task_id,
        title=title.strip() or "Untitled task",
        description=description.strip(),
        duration_hours=max(duration, 0.0),
        quadrant=quadrant,
        position=next_position,
        created_at=datetime.utcnow().isoformat(),
    )


def format_task_card(task: Task) -> str:
    duration = format_duration(task.duration_hours)
    pieces = [task.title.strip() or "Untitled task", f"{duration}h"]
    if task.description.strip():
        pieces.append(task.description.strip())
    # Use unique delimiter for lookup without cluttering UI
    display = " — ".join(pieces)
    return f"{display} ⧉{task.id}"


def parse_task_id_from_card(card_label: str) -> Optional[str]:
    if "⧉" not in card_label:
        return None
    return card_label.split("⧉", 1)[-1].strip()


def format_duration(hours: float) -> str:
    if hours.is_integer():
        return str(int(hours))
    return f"{hours:.2f}".rstrip("0").rstrip(".")


def build_board_payload(tasks: List[Task]) -> List[Dict[str, List[str]]]:
    board_data: List[Dict[str, List[str]]] = []
    for quadrant in PRIORITY_ORDER:
        meta = QUADRANTS[quadrant]
        quadrant_tasks = sorted(
            [task for task in tasks if task.quadrant == quadrant],
            key=lambda task: (task.position, task.created_at),
        )
        items = [format_task_card(task) for task in quadrant_tasks]
        board_data.append({
            "header": f"{meta['header']}\n{meta['subtitle']}",
            "items": items,
        })
    return board_data


def sync_board_state(board_state: List[Dict]) -> None:
    tasks_by_id = {task.id: task for task in st.session_state.tasks}
    for container in board_state:
        header = container.get("header", "")
        quadrant = HEADER_TO_QUADRANT.get(header.split("\n")[0])
        if not quadrant:
            continue
        for index, label in enumerate(container.get("items", [])):
            task_id = parse_task_id_from_card(label)
            if not task_id:
                continue
            task = tasks_by_id.get(task_id)
            if not task:
                continue
            if task.quadrant != quadrant or task.position != index:
                task.quadrant = quadrant
                task.position = index


def compute_time_allocation(tasks: List[Task], available_hours: float, break_minutes: float) -> Dict[str, float]:
    break_hours = max(break_minutes / 60.0, 0.0)
    effective_hours = max(available_hours - break_hours, 0.0)
    remaining = effective_hours
    allocation: Dict[str, float] = {quadrant: 0.0 for quadrant in PRIORITY_ORDER}
    for quadrant in PRIORITY_ORDER:
        required = sum(task.duration_hours for task in tasks if task.quadrant == quadrant)
        if remaining <= 0:
            allocation[quadrant] = 0.0
            continue
        allocated = min(required, remaining)
        allocation[quadrant] = round(allocated, 4)
        remaining -= allocated
    allocation["Break"] = round(break_hours, 4)
    total_allocated = sum(value for key, value in allocation.items() if key != "Break")
    allocation["Buffer"] = round(max(available_hours - allocation["Break"] - total_allocated, 0.0), 4)
    return allocation


def schedule_tasks(
    tasks: List[Task],
    available_hours: float,
    break_minutes: float,
    focus_block_minutes: float,
    session_start: datetime,
) -> Dict[str, List]:
    break_hours = max(break_minutes / 60.0, 0.0)
    effective_hours = max(available_hours - break_hours, 0.0)
    session_end = session_start + timedelta(hours=available_hours)
    remaining_break_minutes = max(break_minutes, 0.0)
    focus_block = max(focus_block_minutes, 15.0)

    rows: List[Dict] = []
    unscheduled: List[Task] = []
    current_time = session_start
    work_since_break = 0.0

    def insert_break(length_minutes: float) -> None:
        nonlocal current_time, remaining_break_minutes, work_since_break
        if length_minutes <= 0:
            return
        break_length = min(length_minutes, remaining_break_minutes)
        if break_length <= 0:
            return
        break_start = current_time
        break_end = break_start + timedelta(minutes=break_length)
        if break_end > session_end:
            break_end = session_end
        rows.append({
            "Task": "Break",
            "Quadrant": "Break",
            "Start": break_start,
            "Finish": break_end,
            "Duration": break_length / 60.0,
        })
        current_time = break_end
        remaining_break_minutes -= break_length
        work_since_break = 0.0

    for quadrant in PRIORITY_ORDER:
        quadrant_tasks = sorted(
            [task for task in tasks if task.quadrant == quadrant and task.duration_hours > 0],
            key=lambda task: (task.position, task.created_at),
        )
        for task in quadrant_tasks:
            if current_time >= session_start + timedelta(hours=effective_hours + break_hours):
                unscheduled.append(task)
                continue
            available_time = (session_end - current_time).total_seconds() / 3600.0
            if available_time <= 0:
                unscheduled.append(task)
                continue
            remaining_task_hours = task.duration_hours
            if work_since_break >= focus_block / 60.0 and remaining_break_minutes > 0:
                insert_break(min(remaining_break_minutes, focus_block / 4.0))
            work_block = min(remaining_task_hours, available_time)
            if work_block <= 0:
                unscheduled.append(task)
                continue
            start_time = current_time
            finish_time = start_time + timedelta(hours=work_block)
            rows.append({
                "Task": task.title,
                "Quadrant": quadrant,
                "Start": start_time,
                "Finish": finish_time,
                "Duration": work_block,
                "Description": task.description,
            })
            current_time = finish_time
            remaining_task_hours -= work_block
            work_since_break += work_block
            if remaining_task_hours > 0:
                clone = Task(
                    id=f"{task.id}-remaining",
                    title=f"{task.title} (continue)",
                    description=task.description,
                    duration_hours=remaining_task_hours,
                    quadrant=task.quadrant,
                    position=task.position,
                    created_at=task.created_at,
                )
                unscheduled.append(clone)
            if remaining_break_minutes > 0 and work_since_break >= focus_block / 60.0:
                insert_break(min(remaining_break_minutes, focus_block / 4.0))

    if remaining_break_minutes > 0 and current_time < session_end:
        insert_break(remaining_break_minutes)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("Start", inplace=True)
    return {"schedule": df, "unscheduled": unscheduled}


def render_pie_chart(allocation: Dict[str, float]) -> None:
    usable = {label: hours for label, hours in allocation.items() if hours > 0}
    if not usable:
        st.info("Add durations and available time to see prioritisation insights.")
        return
    fig = px.pie(
        names=list(usable.keys()),
        values=list(usable.values()),
        title="Time Allocation Overview",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)


def render_gantt(schedule_df: pd.DataFrame) -> None:
    if schedule_df.empty:
        st.warning("Plan a work session to generate a Gantt chart.")
        return
    fig = px.timeline(
        schedule_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Quadrant",
        hover_data={"Description": True, "Duration": ":.2f"},
        color_discrete_map={key: meta["color"] for key, meta in QUADRANTS.items()},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


def load_payload(payload: Dict) -> None:
    tasks_raw = payload.get("tasks") or []
    settings_raw = payload.get("settings") or {}
    parsed_tasks: List[Task] = [Task.from_dict(task) for task in tasks_raw]
    st.session_state.tasks = parsed_tasks
    for key in ("available_hours", "break_minutes", "focus_block_minutes"):
        if key in settings_raw:
            try:
                st.session_state.settings[key] = float(settings_raw[key])
            except (TypeError, ValueError):
                continue
    session_start = settings_raw.get("session_start")
    if session_start:
        try:
            st.session_state.settings["session_start"] = datetime.fromisoformat(session_start)
        except ValueError:
            pass


def build_snapshot() -> str:
    payload = {
        "tasks": [task.to_serializable() for task in st.session_state.tasks],
        "settings": {
            "available_hours": st.session_state.settings["available_hours"],
            "break_minutes": st.session_state.settings["break_minutes"],
            "focus_block_minutes": st.session_state.settings["focus_block_minutes"],
            "session_start": st.session_state.settings["session_start"].isoformat(),
        },
    }
    return json.dumps(payload, indent=2)


def render_task_management() -> None:
    st.subheader("Add a task")
    with st.form("task_form", clear_on_submit=True):
        title = st.text_input("Title", placeholder="Draft proposal")
        description = st.text_area("Description", placeholder="Add relevant notes, resources, or acceptance criteria")
        duration = st.number_input("Duration (hours)", min_value=0.0, step=0.25, value=1.0)
        quadrant_choice = st.selectbox(
            "Initial quadrant",
            options=[(key, f"{meta['header']} · {meta['subtitle']}") for key, meta in QUADRANTS.items()],
            format_func=lambda item: item[1],
        )
        submitted = st.form_submit_button("Add task")
    if submitted:
        task = make_task(title, description, duration, quadrant_choice[0])
        st.session_state.tasks.append(task)
        st.success(f"Added task: {task.title}")

    if st.session_state.tasks:
        with st.expander("Manage tasks"):
            for task in list(st.session_state.tasks):
                cols = st.columns([3, 2, 1])
                cols[0].markdown(f"**{task.title}**")
                cols[1].markdown(f"{format_duration(task.duration_hours)}h · {QUADRANTS[task.quadrant]['header']}")
                if cols[2].button("Remove", key=f"remove_{task.id}"):
                    st.session_state.tasks = [item for item in st.session_state.tasks if item.id != task.id]
                    st.experimental_rerun()


def render_board(tasks: List[Task]) -> None:
    st.subheader("Eisenhower Matrix")
    if not tasks:
        st.info("Add tasks to begin arranging them by importance and urgency.")
        return
    board_data = build_board_payload(tasks)
    sorted_board = sort_items(
        board_data,
        multi_containers=True,
        direction="vertical",
        custom_style=STYLE_BOARD,
        key="eisenhower_board",
    )
    if sorted_board:
        sync_board_state(sorted_board)


def render_insights(tasks: List[Task]) -> None:
    st.subheader("Planning Insights")
    available_hours = st.session_state.settings["available_hours"]
    break_minutes = st.session_state.settings["break_minutes"]
    focus_block = st.session_state.settings["focus_block_minutes"]
    session_start = st.session_state.settings["session_start"]

    total_hours = sum(task.duration_hours for task in tasks)
    st.metric("Total planned task hours", f"{format_duration(total_hours)} h")

    allocation = compute_time_allocation(tasks, available_hours, break_minutes)
    render_pie_chart(allocation)

    planning = schedule_tasks(tasks, available_hours, break_minutes, focus_block, session_start)
    schedule_df: pd.DataFrame = planning["schedule"]
    render_gantt(schedule_df)

    if planning["unscheduled"]:
        st.warning(
            "Some tasks could not be scheduled within the available time. Consider increasing availability or moving lower priority tasks.")
        for leftover in planning["unscheduled"]:
            st.write(f"• {leftover.title} ({format_duration(leftover.duration_hours)} h) [{QUADRANTS.get(leftover.quadrant, {}).get('header', leftover.quadrant)}]")


def render_sidebar() -> None:
    st.sidebar.header("Session Settings")
    settings = st.session_state.settings
    settings["available_hours"] = st.sidebar.number_input(
        "Available hours",
        min_value=0.0,
        max_value=24.0,
        step=0.5,
        value=float(settings["available_hours"]),
    )
    settings["break_minutes"] = st.sidebar.number_input(
        "Total break minutes",
        min_value=0.0,
        max_value=240.0,
        step=5.0,
        value=float(settings["break_minutes"]),
    )
    settings["focus_block_minutes"] = st.sidebar.number_input(
        "Focus block length (minutes)",
        min_value=30.0,
        max_value=240.0,
        step=15.0,
        value=float(settings["focus_block_minutes"]),
    )
    session_start_value = settings["session_start"]
    session_date = st.sidebar.date_input(
        "Session date",
        value=session_start_value.date(),
    )
    session_time = st.sidebar.time_input(
        "Session start time",
        value=session_start_value.time(),
    )
    settings["session_start"] = datetime.combine(session_date, session_time)

    st.sidebar.divider()
    st.sidebar.subheader("Import / Export")
    uploaded = st.sidebar.file_uploader("Upload planner snapshot (.json)", type=["json"])
    if uploaded is not None:
        try:
            payload = json.load(uploaded)
            load_payload(payload)
            st.sidebar.success("Snapshot loaded.")
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file.")

    payload_text = st.sidebar.text_area("Paste snapshot JSON", height=160)
    if st.sidebar.button("Load from text"):
        try:
            payload = json.loads(payload_text)
            load_payload(payload)
            st.sidebar.success("Snapshot loaded from text.")
        except json.JSONDecodeError as exc:
            st.sidebar.error(f"Could not parse JSON: {exc}")

    snapshot = build_snapshot()
    st.sidebar.download_button(
        label="Download current snapshot",
        file_name="eisenhower_snapshot.json",
        mime="application/json",
        data=snapshot,
    )

    with st.sidebar.expander("Snapshot schema"):
        example = {
            "tasks": [
                {
                    "id": "uuid-string",
                    "title": "Prepare kickoff deck",
                    "description": "Outline objectives and assign presenters",
                    "duration_hours": 2.5,
                    "quadrant": "Q1",
                    "position": 0,
                    "created_at": datetime.utcnow().isoformat(),
                }
            ],
            "settings": {
                "available_hours": 8,
                "break_minutes": 45,
                "focus_block_minutes": 120,
                "session_start": datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0).isoformat(),
            },
        }
        st.code(json.dumps(example, indent=2), language="json")


def main() -> None:
    st.set_page_config(
        page_title="Eisenhower Planner",
        layout="wide",
        page_icon="🧭",
    )
    init_session_state()
    render_sidebar()
    st.title("Eisenhower Method Planner")
    st.caption("Organize your work by importance and urgency, plan your day, and keep a reusable snapshot of the plan.")

    render_task_management()
    render_board(st.session_state.tasks)
    render_insights(st.session_state.tasks)


if __name__ == "__main__":
    main()
