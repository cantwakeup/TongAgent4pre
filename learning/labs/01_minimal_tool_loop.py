"""Lesson 1: reproduce one complete model -> tool -> model loop without an API key."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.backends import StateBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def main() -> None:
    """Run a deterministic file-tool loop and verify the final graph state."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/hello.txt",
                                "content": "hello from lesson 1",
                            },
                            "id": "lesson-1-write",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The lesson file has been written."),
            ]
        )
    )

    agent = create_deep_agent(model=model, backend=StateBackend())
    result = agent.invoke({"messages": [HumanMessage(content="Write the lesson file.")]})

    assert result["files"]["/hello.txt"]["content"] == "hello from lesson 1"
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == "The lesson file has been written."

    print("tools exposed to the fake model:")
    print(", ".join(sorted(str(getattr(tool, "name", tool)) for tool in model.tools)))
    print("\nmessage flow:")
    print(" -> ".join(type(message).__name__ for message in result["messages"]))
    print("\nvirtual files:")
    print(result["files"])


if __name__ == "__main__":
    main()
