import time

import supervisely as sly

import globals as g

my_app = sly.AppService()


@my_app.callback("do_something")
@sly.timeit
def do(api: sly.Api, task_id, context, state, app_logger):
    app_logger.info("do something here")
    # or just print (it is recommended to use app_logger)
    # print("do something here")

    steps_count = 10
    task_progress = sly.Progress("Processing...", steps_count)

    for i in range(steps_count):
        print(i)
        time.sleep(1)
        task_progress.iter_done_report()

    my_app.stop()


def main():
    sly.logger.info(
        "Input arguments",
        extra={
            "TASK_ID": g.STATE.task_id,
            "context.teamId": g.STATE.selected_team,
            "context.workspaceId": g.STATE.selected_workspace,
        },
    )

    initial_events = [{"state": None, "context": None, "command": "do_something"}]

    # Run application service
    my_app.run(initial_events=initial_events)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
