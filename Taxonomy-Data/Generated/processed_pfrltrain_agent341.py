import os
import numpy as np

def train_agent(agent, env, steps, outdir, checkpoint_freq=None, max_episode_len=None,
                step_offset=0, evaluator=None, successful_score=None, step_hooks=None,
                eval_during_episode=False, logger=None):
    """
    Train an agent in a given environment for a specified number of steps.

    Parameters:
    - agent: The agent to be trained.
    - env: The environment in which the agent is trained.
    - steps: Total number of steps to train the agent.
    - outdir: Directory to save the agent's model and logs.
    - checkpoint_freq: Frequency of saving the agent's model.
    - max_episode_len: Maximum length of an episode.
    - step_offset: Initial step offset.
    - evaluator: Function to evaluate the agent's performance.
    - successful_score: Score to determine if the training is successful.
    - step_hooks: List of functions to call at each step.
    - eval_during_episode: Whether to evaluate during an episode.
    - logger: Logger for logging training progress.

    Returns:
    - A list of evaluation episode statistics dictionaries.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Initialize variables
    episode_rewards = []
    episode_idx = 0
    eval_stats_history = []

    # Reset the environment
    obs = env.reset()
    episode_reward = 0
    episode_step = 0

    # Training loop
    for step in range(step_offset, steps):
        # Select an action
        action = agent.act(obs)

        # Take the action in the environment
        next_obs, reward, done, info = env.step(action)

        # Update the agent
        agent.observe(obs, action, reward, next_obs, done)

        # Update episode reward and step
        episode_reward += reward
        episode_step += 1

        # Call step hooks if any
        if step_hooks:
            for hook in step_hooks:
                hook(env, agent, step)

        # Check if the episode is done
        if done or (max_episode_len and episode_step >= max_episode_len):
            # Log episode statistics
            logger.info(f'Episode {episode_idx} finished. Reward: {episode_reward}, Steps: {episode_step}')
            episode_rewards.append(episode_reward)

            # Reset environment for the next episode
            obs = env.reset()
            episode_reward = 0
            episode_step = 0
            episode_idx += 1
        else:
            obs = next_obs

        # Evaluate the agent if required
        if evaluator and (eval_during_episode or done):
            eval_stats = evaluator(env, agent)
            eval_stats_history.append(eval_stats)
            logger.info(f'Evaluation at step {step}: {eval_stats}')

            # Check for successful score
            if successful_score is not None and eval_stats.get('mean_reward', 0) >= successful_score:
                logger.info(f'Successful score reached: {eval_stats["mean_reward"]}')
                agent.save(os.path.join(outdir, f'agent_success_{step}.pth'))
                break

        # Save the agent's model at checkpoints
        if checkpoint_freq and step % checkpoint_freq == 0:
            agent.save(os.path.join(outdir, f'agent_checkpoint_{step}.pth'))

    # Save the final model
    agent.save(os.path.join(outdir, 'agent_final.pth'))

    return eval_stats_history