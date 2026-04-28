from dora_ops.classifier import classify_text


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
