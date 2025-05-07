import os
import time
import re
import argparse
import os
import shutil

from AgentOccam.env import DefaultEnviromentWrapper, WebArenaEnvironmentWrapper

from AgentOccam.AgentOccam import AgentOccam
from webagents_step.utils.data_prep import *
from webagents_step.agents.step_agent import StepAgent

from AgentOccam.prompts import AgentOccam_prompt
from webagents_step.prompts.webarena import step_fewshot_template_adapted, step_fewshot_template

from AgentOccam.utils import EVALUATOR_DIR
from playwright.sync_api import sync_playwright

def run():
    parser = argparse.ArgumentParser(
        description="Only the config file argument should be passed"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config file location"
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))
    
    if config.logging:
        if config.logname:
            dstdir = f"{config.logdir}/{config.logname}"
        else:
            dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
    random.seed(42)
    
    config_file_list = []
    
    task_ids = config.env.task_ids
    if hasattr(config.env, "relative_task_dir"):
        relative_task_dir = config.env.relative_task_dir
    else:
        relative_task_dir = "tasks"
    if task_ids == "all" or task_ids == ["all"]:
        task_ids = [filename[:-len(".json")] for filename in os.listdir(f"config_files/{relative_task_dir}") if filename.endswith(".json")]
    for task_id in task_ids:
        config_file_list.append(f"config_files/{relative_task_dir}/{task_id}.json")

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    if config.agent.type == "AgentOccam":
        agent_init = lambda: AgentOccam(
            prompt_dict = {k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
            config = config.agent,
        )
    elif config.agent.type == "AgentOccam-SteP":
            agent_init = lambda: StepAgent(
            root_action = config.agent.root_action,
            action_to_prompt_dict = {k: v for k, v in step_fewshot_template_adapted.__dict__.items() if isinstance(v, dict)},
            low_level_action_list = config.agent.low_level_action_list,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
            )    
    elif config.agent.type == "SteP-replication":
        agent_init = lambda: StepAgent(
            root_action = config.agent.root_action,
            action_to_prompt_dict = {k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)},
            low_level_action_list = config.agent.low_level_action_list,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
        )
    else:
        raise NotImplementedError(f"{config.agent.type} not implemented")

    def handle_dialog(dialog):
        page.dialog_message = dialog.message
        dialog.dismiss()
    
    slow_mo = 1
    viewport_size = {"width": 1920, "height": 1080}
    observation_type="accessibility_tree"

    for config_file in config_file_list:
        with open(config_file, "r") as f:
            task_config = json.load(f)
            print(f"Task {task_config['task_id']}.")
        if os.path.exists(os.path.join(dstdir, f"{task_config['task_id']}.json")):
            print(f"Skip {task_config['task_id']}.")
            continue

        # No need to sleep for 30 mins for Reddit post tasks as alternative way is available to increase the rate limit
        # if task_config['task_id'] in list(range(600, 650))+list(range(681, 689)):
        #     print("Reddit post task. Sleep 30 mins.")
        #     time.sleep(1800)

        # Initialize playwright, page, and context
        context_manager = sync_playwright()
        playwright = context_manager.__enter__()
        browser = playwright.chromium.launch(
            headless=config.env.headless, slow_mo=slow_mo
        )

        storage_state = task_config.get("storage_state", None)
        start_url = task_config.get("start_url", None)
        geolocation = task_config.get("geolocation", None)

        context = browser.new_context(
            viewport=viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )

        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = context.new_page()
                page.on("dialog", handle_dialog)
                client = page.context.new_cdp_session(
                    page
                )  # talk to chrome devtools
                if observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                page.client = client  # type: ignore # TODO[shuyanzh], fix this hackey client
                page.goto(url)
            # set the first page as the current page
            page = context.pages[0]
            page.bring_to_front()
        else:
            page = context.new_page()
            page.on("dialog", handle_dialog)
            client = page.context.new_cdp_session(page)
            if observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            page.client = client  # type: ignore

        """
        env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                        max_browser_rows=config.env.max_browser_rows, 
                                        max_steps=config.max_steps, 
                                        slow_mo=slow_mo, 
                                        observation_type=observation_type, 
                                        current_viewport_only=current_viewport_only, 
                                        viewport_size=viewport_size, 
                                        headless=config.env.headless,
                                        global_config=config,
                                        playwright=playwright,
                                        page=page,
                                        context=context,
                                        context_manager=context_manager)
        """
        env = DefaultEnviromentWrapper(
            objective=task_config["intent"],
            url=start_url,
            max_browser_rows=config.env.max_browser_rows, 
            max_steps=config.max_steps, 
            slow_mo=slow_mo, 
            observation_type=observation_type, 
            current_viewport_only=current_viewport_only, 
            viewport_size=viewport_size, 
            headless=config.env.headless,
            global_config=config,
            playwright=playwright,
            page=page,
            context=context,
            context_manager=context_manager,
        )
        
        agent = agent_init()
        objective = env.get_objective()
        
        # Run AgentOccam agent without ActionEngine compatibility
        # status = agent.act(objective=objective, env=env)

        # Run AgentOccam agent with ActionEngine compatibility
        agent.pre_execute_action(objective, env)
        while(not agent.is_completed(env)):
            status = agent.execute_action(env)

        env.close()

        if config.logging:
            with open(config_file, "r") as f:
                task_config = json.load(f)
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "trajectory": agent.get_trajectory(),
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
            }
            if status:
                summary_data.update(status)
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
    
if __name__ == "__main__":
    run()
