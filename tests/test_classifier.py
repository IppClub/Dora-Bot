from dora_ops.classifier import CLASSIFY_MESSAGE_TOOL, classify_text, classify_text_with_llm


class FakeToolClient:
    def __init__(self, result: dict[str, object] | None = None):
        self.tool = None
        self.tool_name = ""
        self.calls = 0
        self.result = result or {
            "should_accept": False,
            "kind": "project_question",
            "project": "Dora-SSR",
            "confidence": 0.88,
            "needs_repo_analysis": False,
            "summary": "询问渲染管线",
        }

    async def complete_tool_call(self, messages, *, tool, tool_name):
        self.calls += 1
        self.tool = tool
        self.tool_name = tool_name
        return self.result


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


def test_classify_dora_repository_question_routes_to_repo_analysis() -> None:
    result = classify_text("Dora最近的pr有哪些")
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.action == "record_feedback"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_dora_technical_analysis_routes_to_repo_analysis() -> None:
    result = classify_text("分析一下 Dora 的渲染管线怎么实现")
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.action == "record_feedback"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_dora_build_question_routes_to_repo_analysis() -> None:
    result = classify_text("Dora 的 Android 构建为什么失败")
    assert result.should_accept is True
    assert result.kind == "feedback"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_dora_ci_workflow_question_routes_to_repo_analysis() -> None:
    result = classify_text("Dora 的 CI workflow 为啥没跑")
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_dora_coding_agent_concurrency_routes_to_repo_analysis() -> None:
    result = classify_text("@多萝 分析一下dora coding agent 如何处理并发工具的")
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_yue_moon_script_question_routes_to_repo_analysis() -> None:
    result = classify_text("Yue 的 moon script 兼容逻辑在哪")
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "YueScript"
    assert result.needs_repo_analysis is True


def test_classify_project_fault_language_routes_to_repo_analysis() -> None:
    result = classify_text("多萝，Dora SSR 资源释放有问题，场景切换后显存没下来")
    assert result.should_accept is True
    assert result.kind == "feedback"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


def test_classify_generic_technical_topic_does_not_infer_project() -> None:
    result = classify_text("WASM 的渲染管线怎么拆比较稳")
    assert result.should_accept is False
    assert result.kind == "chat"
    assert result.project is None
    assert result.needs_repo_analysis is False


async def test_llm_classifier_trusts_repo_analysis_when_keywords_miss() -> None:
    client = FakeToolClient()

    result = await classify_text_with_llm("这个帧管线怎么拆", client)  # type: ignore[arg-type]

    assert client.tool == CLASSIFY_MESSAGE_TOOL
    assert client.tool_name == "classify_message"
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.action == "record_feedback"
    assert result.needs_repo_analysis is True


async def test_llm_classifier_skips_llm_when_keywords_trigger_repo_analysis() -> None:
    client = FakeToolClient()

    result = await classify_text_with_llm("Dora 的 CI workflow 为啥没跑", client)  # type: ignore[arg-type]

    assert client.calls == 0
    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.needs_repo_analysis is True


async def test_llm_classifier_can_use_context_project_anchor() -> None:
    client = FakeToolClient(
        {
            "should_accept": True,
            "kind": "project_question",
            "project": "Dora-SSR",
            "confidence": 0.88,
            "needs_repo_analysis": True,
            "summary": "追问是否需要分析",
        }
    )

    result = await classify_text_with_llm(
        "这个要不要跑仓库分析？",
        client,  # type: ignore[arg-type]
        context_text="tester(QQ:789)：Dora SSR 的 Web IDE 创建文件后无法刷新",
    )

    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.action == "record_feedback"
    assert result.needs_repo_analysis is True


async def test_llm_classifier_forces_repo_analysis_for_context_followup() -> None:
    client = FakeToolClient(
        {
            "should_accept": False,
            "kind": "chat",
            "action": "ignore",
            "project": None,
            "confidence": 0.95,
            "needs_repo_analysis": False,
            "summary": "用户询问是否需要仓库分析",
        }
    )

    result = await classify_text_with_llm(
        "这个问题要不要跑仓库分析？",
        client,  # type: ignore[arg-type]
        context_text="tester(QQ:789)：Dora SSR Web IDE 创建文件后无法刷新",
    )

    assert result.should_accept is True
    assert result.kind == "project_question"
    assert result.project == "Dora-SSR"
    assert result.action == "record_feedback"
    assert result.needs_repo_analysis is True
