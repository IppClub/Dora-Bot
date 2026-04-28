from dora_ops.classifier import CLASSIFY_MESSAGE_TOOL, classify_text, classify_text_with_llm


class FakeToolClient:
    def __init__(self):
        self.tool = None
        self.tool_name = ""

    async def complete_tool_call(self, messages, *, tool, tool_name):
        self.tool = tool
        self.tool_name = tool_name
        return {
            "should_accept": False,
            "kind": "project_question",
            "project": "Dora-SSR",
            "confidence": 0.88,
            "needs_repo_analysis": False,
            "summary": "询问渲染管线",
        }


def test_classify_dora_feedback() -> None:
    result = classify_text("Dora SSR 的 Web IDE 创建文件后无法刷新")
    assert result.should_accept is True
    assert result.kind == "feedback"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_unrelated_chat() -> None:
    result = classify_text("今天晚上吃什么")
    assert result.should_accept is False
    assert result.kind == "chat"


def test_classify_project_question_routes_to_repo_analysis() -> None:
    result = classify_text("Dora SSR 渲染管线怎么拆比较稳")
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.action == "record_feedback"
    assert result.needs_repo_analysis is True


async def test_llm_classifier_uses_function_calling() -> None:
    client = FakeToolClient()

    result = await classify_text_with_llm("这个帧管线怎么拆", client)  # type: ignore[arg-type]

    assert client.tool == CLASSIFY_MESSAGE_TOOL
    assert client.tool_name == "classify_message"
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.action == "record_feedback"
    assert result.needs_repo_analysis is True
