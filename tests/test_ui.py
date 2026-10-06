from pathlib import Path

from scripts import ui

ROOT = Path(__file__).resolve().parents[1]


class _UnusedAssistant:
    def handle(self, *_args, **_kwargs):  # pragma: no cover - UI construction only
        raise AssertionError("the assistant must not run while building the UI")


def test_prompt_is_explicitly_editable_and_rtl():
    demo = ui.create_demo(_UnusedAssistant())
    config = demo.get_config_file()
    prompt = next(
        component
        for component in config["components"]
        if component["props"].get("elem_id") == "ppq-prompt"
    )
    props = prompt["props"]

    assert props["interactive"] is True
    assert props["rtl"] is True
    assert props["text_align"] == "right"
    assert props["max_length"] == 2000


def test_runtime_text_is_html_escaped():
    rendered = ui._answer_body_html('<script>alert("x")</script>')
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_gradio_6_styling_is_passed_at_launch(monkeypatch):
    captured = {}

    class Demo:
        def queue(self, **kwargs):
            captured["queue"] = kwargs
            return self

        def launch(self, **kwargs):
            captured["launch"] = kwargs

    monkeypatch.setattr(ui, "_theme", lambda: "test-theme")
    ui.launch_demo(Demo(), port=7788, share=False)

    assert captured["queue"] == {"default_concurrency_limit": 1}
    assert captured["launch"]["server_port"] == 7788
    assert captured["launch"]["theme"] == "test-theme"
    assert captured["launch"]["css"] == ui.UI_CSS
    assert captured["launch"]["head"] == ui.UI_HEAD
