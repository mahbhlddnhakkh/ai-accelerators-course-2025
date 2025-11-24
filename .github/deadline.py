import os
import sys
from github import Github
from datetime import datetime
from zoneinfo import ZoneInfo


def main():
    moscow_tz = ZoneInfo("Europe/Moscow")
    current_date = datetime.now(moscow_tz)
    deadline_date = {
        "01-softmax-cpu": datetime(2025, 11, 21, hour=1, tzinfo=moscow_tz),
        "02-softmax-cuda": datetime(2025, 11, 21, hour=1, tzinfo=moscow_tz),
        "03-matmul-cuda": datetime(2025, 11, 21, hour=1, tzinfo=moscow_tz),
        "04-softmax-ascend": datetime(2025, 12, 19, hour=1, tzinfo=moscow_tz),
        "05-mixed-ascend": datetime(2025, 12, 26, hour=1, tzinfo=moscow_tz),
    }

    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(os.environ["GITHUB_REPOSITORY"])
    pr = repo.get_pull(int(os.environ["PR_NUMBER"]))
    labels = list(deadline_date.keys())
    lab_label = None

    for label in pr.get_labels():
        if label.name in labels:
            lab_label = label.name

    if not lab_label:
        return

    if current_date > deadline_date[lab_label]:
        pr.add_to_labels("delayed")


if __name__ == "__main__":
    sys.exit(main())
