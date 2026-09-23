"""Run LeRobot evaluation with the ALOHA dataset instruction."""

import gym_aloha.env


gym_aloha.env.AlohaEnv.task_description = "Insert the peg into the socket."

from lerobot.scripts.lerobot_eval import main


if __name__ == "__main__":
    main()
