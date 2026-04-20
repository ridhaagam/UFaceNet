from ufacenet.tasks import TASK_REGISTRY, normalize_task_request


def test_all_tasks_include_frec_and_recognition():
    tasks = normalize_task_request("all")
    assert "frec" in tasks
    assert "recognition" in tasks
    assert TASK_REGISTRY["recognition"].short_name == "FR"
    assert TASK_REGISTRY["frec"].short_name == "FRec"


def test_aliases_are_stable():
    assert normalize_task_request("analysis")[0] == "parsing"
    assert normalize_task_request("reconstruction") == ("frec",)
