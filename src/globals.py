import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    # * For convinient development, has no effect in the production.
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


# * To avoid global variables in different modules, it's better to use g.STATE (g.AppState) object
# * across the app. It can be accessed from any module by importing globals module.
class State:
    def __init__(self):
        # * This class should contain all the variables that are used across the app.
        # * For example selected team, workspace, project, dataset, etc.
        self.task_id = sly.env.task_id()
        self.selected_team = sly.env.team_id()
        self.selected_workspace = sly.env.workspace_id()

        # * If you need to get the project and dataset from the environment variables,
        # * you can uncomment the following lines.
        # self.selected_project = sly.env.project_id(raise_not_found=False)
        # self.selected_dataset = sly.env.dataset_id(raise_not_found=False)

        self.continue_working = True


# * Class object to access from other modules.
# * import src.globals as g
# * selected_team = g.STATE.selected_team
STATE = State()
