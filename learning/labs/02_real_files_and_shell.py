"""Lesson 1B: run deterministic file and shell tools against the real host."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.backends import LocalShellBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def main() -> None:
    """Write a real file, inspect it with a real shell, and verify both results."""
    runtime_dir = Path(__file__).resolve().parents[1] / ".runtime" / "lesson-01-real"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/real.txt",
                                "content": "this line exists on the real disk",
                            },
                            "id": "lesson-1b-write",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {"command": "pwd && ls -l real.txt && sed -n '1p' real.txt"},
                            "id": "lesson-1b-execute",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The real file and shell checks passed."),
            ]
        )
    )

    backend = LocalShellBackend(
        root_dir=runtime_dir,
        virtual_mode=True,
        timeout=5,
        env={"PATH": "/usr/bin:/bin"},
        inherit_env=False,
    )
    agent = create_deep_agent(model=model, backend=backend)
    result = agent.invoke({"messages": [HumanMessage(content="Run the real-backend lesson.")]})

    real_file = runtime_dir / "real.txt"
    assert real_file.read_text() == "this line exists on the real disk"

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    execute_result = next(message for message in tool_messages if message.tool_call_id == "lesson-1b-execute")
    assert str(runtime_dir) in str(execute_result.content)
    assert "this line exists on the real disk" in str(execute_result.content)
    assert "execute" in {str(getattr(tool, "name", tool)) for tool in model.tools}

    print(f"real file: {real_file}")
    print(f"real content: {real_file.read_text()}")
    print("\nexecute ToolMessage:")
    print(execute_result.content)
    print("\nmessage flow:")
    print(" -> ".join(type(message).__name__ for message in result["messages"]))


if __name__ == "__main__":
    main()
